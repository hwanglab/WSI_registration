"""
Author: Minji Kim  
Creation date: 10/24/2022
"""
# import pkgs
import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Dict


def prepare_transform_dict(deformField_file_path: str = None) -> Dict:
    """
    This function is for building transformation dictionary from deformField file.

     Args:
        deformField_file_path (str): file path of deformField.csv file. Defaults to None.

    Returns:
        transform_dict (Dict): transformation dictionary for mapping.
    """

    # read deformField file
    deformed_df = pd.read_csv(deformField_file_path)

    # drop null values
    deformed_df.dropna(inplace=True)

    # convert data type into int
    deformed_df = deformed_df.astype(
        {"source_x": "int", "source_y": "int", "target_x": "int", "target_y": "int"}
    )

    # prepare source coordinates and target coordinates
    source_coords = list(zip(deformed_df["source_x"], deformed_df["source_y"]))
    target_coords = list(zip(deformed_df["target_x"], deformed_df["target_y"]))

    # prepare transformation dictionary keys: source coordinates, values: target coordinates
    transform_dict = dict(zip(source_coords, target_coords))
    return transform_dict


def map_coords(
    source_df: pd.DataFrame, target_df: pd.DataFrame, transform_dict: Dict
) -> pd.DataFrame:

    """
    This function is for mapping coordinates between source and target.

    Args:
        source_df (pd.DataFrame): source tile file information
        target_df (pd.DataFrame): target tile file information
        transform_dict (Dict): transformation dictionary

    Returns:
        merge_df: (pd.DataFrame): mapping result
    """

    # before mapping, modify target thumbnail coordinates by substracting offset values
    source_df["source_mask_coords"] = list(
        map(tuple, zip(source_df["source_mask_x"], source_df["source_mask_y"]))
    )

    # mapping target thumbnail coordinate and source coordincates
    source_df["target_mask_coords"] = source_df["source_mask_coords"].apply(
        lambda x: transform_dict.get(x)
    )

    # before mapping, modify target thumbnail coordinates by substracting offset values
    target_df["target_mask_coords"] = list(
        map(tuple, zip(target_df["target_mask_x"], target_df["target_mask_y"]))
    )

    merge_df = source_df.merge(target_df, on="target_mask_coords", how="left")
    merge_df.drop(columns=["source_mask_coords", "target_mask_coords"], inplace=True)

    return merge_df


def main(args):

    deformField_paths = Path(args.data_root_path).rglob("deformField.csv")

    for deformField_path in deformField_paths:

        # get parts of deformField_path
        file_path_parts = deformField_path.parts

        # trim information (image_id, source_file_name, target_file_name)
        image_id = file_path_parts[1]
        file_name = file_path_parts[2]
        file_name_lst = file_name.split("_")
        source_file_name = f"{file_name_lst[1]}_{file_name_lst[2]}"
        target_file_name = f"{file_name_lst[1]}_{file_name_lst[3]}"

        # build transformation dictionary for mapping
        transform_dict = prepare_transform_dict(deformField_path)

        # prepare source and target file path
        source_file_path = list(
            Path(f"{args.data_root_path}/{image_id}").glob(f"*{source_file_name}*.tsv")
        )[0]
        target_file_path = list(
            Path(f"{args.data_root_path}/{image_id}").glob(f"*{target_file_name}*.tsv")
        )[0]

        # read source and target information file
        source_df = pd.read_csv(source_file_path, sep="\t")
        target_df = pd.read_csv(target_file_path, sep="\t")

        # drop non-tile rows
        source_df = source_df[source_df.filename.notnull()]
        target_df = target_df[target_df.filename.notnull()]

        # reset index
        source_df.reset_index(inplace=True, drop=True)
        target_df.reset_index(inplace=True, drop=True)

        # add prefix for each dataframe
        source_df = source_df.add_prefix("source_")
        target_df = target_df.add_prefix("target_")

        # create mapped dataframe
        merge_df = map_coords(source_df, target_df, transform_dict)

        save_path = f"{args.save_path}/{image_id}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save mapped dataframe into csv file
        merge_df_filename = (
            f"{save_path}/{image_id}_{source_file_name}_{file_name_lst[3]}.csv"
        )
        merge_df.to_csv(merge_df_filename, index=False)


if __name__ == "__main__":

    # Generate settings
    parser = argparse.ArgumentParser(
        description="Configurations for WSI coordinates mapping"
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="Van_Abel_all_thumbnail_images",
        help="data directory",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Van_Abel_mapped_information",
        help="result directory",
    )
    args = parser.parse_args()

    main(args)
