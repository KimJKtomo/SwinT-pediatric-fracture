import os
import re
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from ao_to_salter_utils import extract_ao_subtypes

def get_dataset_paths(base_path, part):
    if part == 'radius':
        return os.path.join(base_path, 'segmentation_frac', 'radius_set', 'crop_set', 'train')
    elif part == 'scaphoid':
        return os.path.join(base_path, 'scaphoid_torch', 'train_torch', 'train_img', 'positive')
    elif part == 'styloid':
        return os.path.join(base_path, 'styloid_torch', 'train_torch', 'img')
    else:
        raise ValueError(f"Unknown part: {part}")

def extract_part(ao_code):
    if pd.isna(ao_code):
        return 'all'
    ao_code = str(ao_code).lower()
    if '23r' in ao_code:
        return 'radius'
    elif '23u' in ao_code:
        return 'ulna'
    elif '23m' in ao_code or '-m' in ao_code:
        return 'metaphyseal'
    elif 'scaphoid' in ao_code:
        return 'scaphoid'
    else:
        return 'all'

def age_group(age):
    try:
        age = int(age)
        if age <= 5:
            return '0-5'
        elif age <= 10:
            return '6-10'
        elif age <= 15:
            return '11-15'
        elif age <= 20:
            return '16-20'
        else:
            return '21+'
    except:
        return 'unknown'
def load_mura():
    image_csv_path = "/mnt/data/KimJG/ELBOW_test/MURA-v1.1/train_image_paths.csv"
    label_csv_path = "/mnt/data/KimJG/ELBOW_test/MURA-v1.1/train_labeled_studies.csv"

    # 이미지 경로 불러오기
    train_img_df = pd.read_csv(image_csv_path, header=None, names=["path"])
    train_img_df['path'] = train_img_df['path'].str.strip()
    train_img_df['study_id'] = train_img_df['path'].apply(lambda x: '/'.join(x.split('/')[:7]))

    # 라벨 불러오기
    train_label_df = pd.read_csv(label_csv_path, header=None, names=["path", "label"])
    train_label_df['path'] = train_label_df['path'].str.strip('/')
    train_label_df['study_id'] = train_label_df['path']

    # 병합
    train_df = pd.merge(train_img_df, train_label_df[['study_id', 'label']], on="study_id", how="left")
    train_df = train_df.dropna(subset=["label"])
    train_df["label"] = train_df["label"].astype(int)

    # 이미지 파일 실제 경로 설정
    train_df["image_path"] = train_df["path"].apply(lambda x: os.path.join("/mnt/data/KimJG/ELBOW_test/MURA-v1.1", x + ".png"))

    train_df["part"] = "wrist"
    train_df["age_group"] = "21+"
    train_df["fracture_visible"] = train_df["label"]
    train_df["ao_classification"] = None
    train_df["ao_subtypes"] = [[] for _ in range(len(train_df))]
    train_df["ao_primary"] = "Unknown"
    train_df["gender"] = "U"
    train_df["source"] = "mura"
    train_df["split"] = "train"

    return train_df[[
        "image_path", "label", "part", "age_group", "fracture_visible",
        "ao_classification", "ao_subtypes", "ao_primary", "gender", "source", "split"
    ]]


def get_combined_dataset(image_only=True, fracture_only=True):
    kaggle_path = "/mnt/data/KimJG/ELBOW_test/Kaggle_dataset"
    kaggle_csv = os.path.join(kaggle_path, "dataset.csv")
    df = pd.read_csv(kaggle_csv)

    df['image_path'] = df['filestem'].apply(lambda f: next(iter(glob(os.path.join(kaggle_path, 'images_part*', f"{f}.*"))), None))
    df = df[df['image_path'].notnull()]
    df = df[(df['metal'].isna()) | (df['metal'] != 1)]
    df['part'] = df['ao_classification'].apply(extract_part)
    df['age_group'] = df['age'].apply(age_group)
    df = df[df['age_group'] != 'unknown']
    df['label'] = df['fracture_visible'].apply(lambda x: 1 if pd.notna(x) else 0)
    df['fracture_visible'] = df['label']
    df['source'] = 'open'
    df['ao_subtypes'] = df['ao_classification'].apply(extract_ao_subtypes)
    df['ao_primary'] = df['ao_subtypes'].apply(lambda x: x[0] if x else 'Unknown')
    df['split'] = df['diagnosis_uncertain'].apply(lambda x: 'test' if x == 1.0 else 'train_val')

    test_df = df[df['split'] == 'test']
    train_val_df = df[df['split'] == 'train_val']
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])
    train_df['split'], val_df['split'] = 'train', 'val'
    kaggle_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 병원 데이터
    hospital_paths, hospital_labels, hospital_parts = [], [], []
    hospital_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/train_set"
    for part in ['radius', 'scaphoid', 'styloid']:
        part_path = get_dataset_paths(hospital_path, part)
        if 'segmentation_frac' in part_path:
            for cname, label in [('negative', 0), ('positive', 1)]:
                files = glob(os.path.join(part_path, cname, '*'))
                hospital_paths += files
                hospital_labels += [label] * len(files)
                hospital_parts += [part] * len(files)
        else:
            files = glob(os.path.join(part_path, '*'))
            for f in files:
                fname = os.path.basename(f).lower()
                if 'posi' in fname:
                    label = 1
                elif 'nega' in fname:
                    label = 0
                else:
                    continue
                hospital_paths.append(f)
                hospital_labels.append(label)
                hospital_parts.append(part)

    hospital_df = pd.DataFrame({
        'image_path': hospital_paths,
        'label': hospital_labels,
        'part': hospital_parts,
        'age_group': '21+',
        'fracture_visible': hospital_labels,
        'ao_classification': None,
        'ao_subtypes': [[] for _ in range(len(hospital_paths))],
        'ao_primary': 'Unknown',
        'gender': 'F',
        'source': 'clinical',
        'split': 'train'
    })

    # MURA 데이터 로드
    mura_df = load_mura()

    combined_df = pd.concat([
        kaggle_df[['image_path', 'label', 'part', 'age_group', 'fracture_visible',
                   'ao_classification', 'ao_subtypes', 'ao_primary', 'gender', 'source', 'split']],
        hospital_df,
        mura_df
    ], ignore_index=True)

    if image_only:
        combined_df = combined_df[combined_df['image_path'].notnull()]
    if fracture_only:
        combined_df = combined_df[combined_df['label'].isin([0, 1])]

    return combined_df
