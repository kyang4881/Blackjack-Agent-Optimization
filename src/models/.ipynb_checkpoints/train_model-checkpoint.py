from Fedspeak.src.models.preprocessing import preprocessor


class trainPipeline(preprocessor):
    """A class for computing evaluation metrics and training the model
    Args:
        model (obj): A pre-trained model
        repository_id (str): A string id for the repository
        learning_rate (float): The initial learning rate for AdamW optimizer
        per_device_train_batch_size (int): The batch size per GPU/TPU core/CPU for training
        per_device_eval_batch_size (int): The batch size per GPU/TPU core/CPU for evaluation
        weight_decay (float): The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
        save_total_limit (int): If a value is passed, will limit the total amount of checkpoints
        num_train_epochs (int): Number of epoch to train
        data_dict (dataset): A dataset containing the train, validation, and test data
        padding (bool): A boolean or string for specifying the padding requirement
        truncation (bool): A boolean or string for specifying the truncation requirement
        tokenizer (obj): A transformer object
        tokenized_dict (dataset): A dataset containing tokenized train, validation, and test data
        evaluation_strategy (str): The evaluation strategy to adopt during training
        save_strategy (str): The checkpoint save strategy to adopt during training
        load_best_model_at_end (bool): Whether or not to load the best model found during training at the end of training
        logging_strategy (str): The logging strategy to adopt during training
        logging_steps (int): Number of update steps between two logs
        overwrite_output_dir (bool): If True, overwrite the content of the output directory
        device (obj): Specifies whether to use cpu or gpu
        metric_for_best_model (str): What type of metric to use for selecting the best model
        greater_is_better (bool): Defines whether greater is better in the metric for the best model
        seed (int): A random seed

    """
    def __init__(
        self,
        model,
        repository_id,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        weight_decay,
        save_total_limit,
        num_train_epochs,
        data_dict,
        padding,
        truncation,
        tokenizer,
        tokenized_dict,
        evaluation_strategy,
        save_strategy,
        load_best_model_at_end,
        logging_strategy,
        logging_steps,
        overwrite_output_dir,
        device,
        metric_for_best_model,
        greater_is_better,
        seed

    ):
        super().__init__(data_dict, padding, truncation, tokenizer)

        self.model = model
        self.repository_id = repository_id
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.save_total_limit = save_total_limit
        self.num_train_epochs = num_train_epochs
        self.fp16 = False
        self.predict_with_generate = True
        self.skip_special_tokens = True
        self.output_dir = self.repository_id
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.tokenized_dict = tokenized_dict
        self.logging_dir = self.repository_id  + "/logs"
        self.logging_strategy = logging_strategy
        self.logging_steps = logging_steps
        self.overwrite_output_dir = overwrite_output_dir
        self.device = device
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.seed = seed

    def compute_metrics(self, eval_preds):
        """Compute the evaluation metrics
        Args:
            eval_preds(arr): predictions and labels
        Returns:
            Evaluation results
        """
        #metric = evaluate.load("rouge")
        preds, labels = eval_preds
        # Remove the -100
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode the predictions and true labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=self.skip_special_tokens)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=self.skip_special_tokens)
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        # Comptue the evaluation metric
        try:
            acc = accuracy_score(decoded_labels, decoded_preds)
        except:
            print(decoded_labels)
            print(decoded_preds)
        return {'accuracy': acc}


    def training(self):
        """A method for training the model
        Returns:
            The trained model
        """
        # Defining the data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        # Defining the training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=self.evaluation_strategy,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            weight_decay=self.weight_decay,
            save_total_limit=self.save_total_limit,
            num_train_epochs=self.num_train_epochs,
            fp16=self.fp16,
            predict_with_generate=self.predict_with_generate,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            logging_dir = self.logging_dir,
            logging_strategy = self.logging_strategy,
            logging_steps = self.logging_steps,
            overwrite_output_dir = self.overwrite_output_dir,
            metric_for_best_model = self.metric_for_best_model,
            greater_is_better = self.greater_is_better,
            seed = self.seed#,
            #deepspeed="./ds_config_zero3.json"

        )
        # Defining the trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dict["train"],
            eval_dataset=self.tokenized_dict["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        return trainer