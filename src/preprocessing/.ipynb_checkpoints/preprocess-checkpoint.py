import pandas as pd
import numpy as np
from datasets import DatasetDict, Dataset, concatenate_datasets
import os


class compileData:
    """A class for compiling the necessary json files into a dataset
    Args:
      train_data_fname (str): file name for the train data
      dev_data_fname (str): file name for the validation data
      test_data_fname (str): file name for the test data
      train_data (list): a list containing the train data from the json file
      dev_data (list): a list containing the validation data from the json file
      test_data (list): a list containing the test data from the json file
      prompt (str): A prompt for the model
    """
    def __init__(self, X_train, X_validation, X_test, y_train, y_validation, y_test, prompt):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.prompt = prompt


    def compile_dataset(self):
        """Compile the dataframes into datasets
        Returns:
            A dataset containing the train, validation, and test data, and one that doesn't include the test data
        """
        train_dataset = Dataset.from_dict({
            'original_sentence': self.X_train['sentence'],
            'features': [f"{self.prompt}\n\n{self.X_train['sentence'][i]}\n- {self.X_train['option1'][i]}\n- {self.X_train['option2'][i]}\n- {self.X_train['option3'][i]}\n- {self.X_train['option4'][i]}\n- {self.X_train['option5'][i]}" for i in range(len(self.X_train['sentence']))],
            'option1': self.X_train['option1'],
            'option2': self.X_train['option2'],
            'option3': self.X_train['option3'],
            'option4': self.X_train['option4'],
            'option5': self.X_train['option5'],
            'labels': self.y_train['answer'],
            'labels_int': self.y_train['answer_int']
        })

        dev_dataset = Dataset.from_dict({
            'original_sentence': self.X_validation['sentence'],
            'features': [f"{self.prompt}\n\n{self.X_validation['sentence'][i]}\n- {self.X_validation['option1'][i]}\n- {self.X_validation['option2'][i]}\n- {self.X_validation['option3'][i]}\n- {self.X_validation['option4'][i]}\n- {self.X_validation['option5'][i]}" for i in range(len(self.X_validation['sentence']))],
            'option1': self.X_validation['option1'],
            'option2': self.X_validation['option2'],
            'option3': self.X_validation['option3'],
            'option4': self.X_validation['option4'],
            'option5': self.X_validation['option5'],
            'labels': self.y_validation['answer'],
            'labels_int': self.y_validation['answer_int']
        })

        test_dataset = Dataset.from_dict({
            'original_sentence': self.X_test['sentence'],
            'features': [f"{self.prompt}\n\n{self.X_test['sentence'][i]}\n- {self.X_test['option1'][i]}\n- {self.X_test['option2'][i]}\n- {self.X_test['option3'][i]}\n- {self.X_test['option4'][i]}\n- {self.X_test['option5'][i]}" for i in range(len(self.X_test['sentence']))],
            'option1': self.X_test['option1'],
            'option2': self.X_test['option2'],
            'option3': self.X_test['option3'],
            'option4': self.X_test['option4'],
            'option5': self.X_test['option5'],
            'labels': self.y_test['answer'],
            'labels_int': self.y_test['answer_int']
        })

        nlp_dataset_dict_wtest = DatasetDict({
            'train': train_dataset,
            'validation': dev_dataset,
            'test': test_dataset
        })
        return  nlp_dataset_dict_wtest

class preprocessor:
    """A Preprocessing class for tokenizing the features and labels
    Args:
        data_dict (dataset): A dataset containing train, validation, and test data
        padding (bool/str): A boolean or string for specifying the padding requirement
        truncation (bool): A boolean or string for specifying the truncation requirement
        tokenizer (obj): A transformer object
    """
    def __init__(self, data_dict, padding, truncation, tokenizer):
        self.data_dict = data_dict
        self.padding = padding
        self.truncation = truncation
        self.tokenizer = tokenizer
        tokenized_features = concatenate_datasets([self.data_dict["train"], self.data_dict["validation"], self.data_dict["test"]]).map(lambda x: tokenizer(x["features"], truncation=self.truncation))
        tokenized_labels = concatenate_datasets([self.data_dict["train"], self.data_dict["validation"], self.data_dict["test"]]).map(lambda x: tokenizer(x["labels"], truncation=self.truncation))
        self.max_source_length = max([len(x) for x in tokenized_features["features"]])
        self.max_target_length = max([len(x) for x in tokenized_labels["labels"]])

    def preprocess(self, data):
        """Preprocessing the data by tokenizing the features and labels
        Args:
            data (dataset): A dataset containing the train, validation, and test data
        Returns:
            An updated dataset containing tokenized inputs
        """
        # Tokenize the features
        model_inputs = tokenizer(data["features"], max_length=self.max_source_length, padding=self.padding, truncation=self.truncation)
        # Tokenize the labels
        labels = tokenizer(text_target=data["labels"], max_length=self.max_target_length, padding=self.padding, truncation=self.truncation)
        # For max length padding, replace tokenizer.pad_token_id with -100 to ignore padding in the loss
        if self.padding == "max_length": labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def map_inputs(self):
        """Map the dataset to preprocess the train, validation, and test data
        Returns:
            A preprocessed dataset
        """
        tokenized_dict = self.data_dict.map(self.preprocess, batched=True)
        return tokenized_dict
