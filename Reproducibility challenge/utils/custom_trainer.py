from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.criterion = kwargs.pop('criterion')
        super().__init__(*args, **kwargs)

    # Trainer 클래스의 compute_loss 함수 over-riding
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss