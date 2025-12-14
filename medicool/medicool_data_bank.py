# medicool_data_bank.py
from typing import List, Tuple
from copy import deepcopy

class MedicoolDataBank:

    # -----------------------------
    # Indexing / naming rules
    # -----------------------------
    MINIMUM_DIGIT_COUNT = 3

    INDEX_START_TRAINING = 0
    INDEX_END_TRAINING = 8  # 128

    INDEX_START_TESTING = 0
    INDEX_END_TESTING = 8  # 128

    NAME_BASE_TRAINING = "proto_cells_train_"
    NAME_BASE_TESTING = "proto_cells_test_"   # <-- fixed

    # -----------------------------
    # Directory layout
    # -----------------------------
    SUBDIR_TRAINING_IMAGES = "training_images"
    SUBDIR_TRAINING_LABELS = "training_labels"

    SUBDIR_TESTING_IMAGES = "testing_images"
    SUBDIR_TESTING_LABELS = "testing_labels"

    # -----------------------------
    # File conventions
    # -----------------------------
    LABEL_POSTFIX = "_annotations"
    LABEL_EXTENSION = "json"

    IMAGE_POSTFIX = ""
    IMAGE_EXTENSION = "png"

    # -----------------------------
    # Utilities
    # -----------------------------
    @classmethod
    def number_string(cls, number: int) -> str:
        """Return zero-padded number string using MINIMUM_DIGIT_COUNT."""
        return f"{number:0{cls.MINIMUM_DIGIT_COUNT}d}"

    # -----------------------------
    # Name generation
    # -----------------------------
    @classmethod
    def get_training_names(cls) -> List[str]:
        """Base names only, no postfix or extension."""
        names: List[str] = []
        index = cls.INDEX_START_TRAINING
        while index <= cls.INDEX_END_TRAINING:
            names.append(cls.NAME_BASE_TRAINING + cls.number_string(index))
            index += 1
        return names

    @classmethod
    def get_testing_names(cls) -> List[str]:
        """Base names only, no postfix or extension."""
        names: List[str] = []
        index = cls.INDEX_START_TESTING
        while index <= cls.INDEX_END_TESTING:
            names.append(cls.NAME_BASE_TESTING + cls.number_string(index))
            index += 1
        return names

    # -----------------------------
    # File info
    # -----------------------------
    @classmethod
    def get_training_file_info(cls) -> Tuple[str, str, List[str], List[str]]:
        """
        Returns:
            (images_subdir, labels_subdir, image_file_names, label_file_names)

        image_file_name = base + IMAGE_POSTFIX + "." + IMAGE_EXTENSION
        label_file_name = base + LABEL_POSTFIX + "." + LABEL_EXTENSION
        """
        bases = cls.get_training_names()
        image_file_names: List[str] = []
        label_file_names: List[str] = []

        for base in bases:
            image_file_names.append(base + cls.IMAGE_POSTFIX + "." + cls.IMAGE_EXTENSION)
            label_file_names.append(base + cls.LABEL_POSTFIX + "." + cls.LABEL_EXTENSION)

        return (cls.SUBDIR_TRAINING_IMAGES, cls.SUBDIR_TRAINING_LABELS, image_file_names, label_file_names)

    @classmethod
    def get_testing_file_info(cls) -> Tuple[str, str, List[str], List[str]]:
        """
        Returns:
            (images_subdir, labels_subdir, image_file_names, label_file_names)
        """
        bases = cls.get_testing_names()
        image_file_names: List[str] = []
        label_file_names: List[str] = []

        for base in bases:
            image_file_names.append(base + cls.IMAGE_POSTFIX + "." + cls.IMAGE_EXTENSION)
            label_file_names.append(base + cls.LABEL_POSTFIX + "." + cls.LABEL_EXTENSION)

        return (cls.SUBDIR_TESTING_IMAGES, cls.SUBDIR_TESTING_LABELS, image_file_names, label_file_names)
    
    @classmethod
    def append_file_suffix(cls, file_name: str, suffix: str) -> str:

        new_file_name = copy.deepcopy(file_name)
        previous_extension = new_file_name.extension()
        new_file_name.remove_extension()

        new_file_name += suffix

        if previous_extension:
            new_file_name += "." + previous_extension
        
        return new_file_name




