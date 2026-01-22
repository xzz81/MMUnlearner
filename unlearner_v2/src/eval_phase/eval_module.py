import torch
import pytorch_lightning as pl
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re


class EvalLightningModule(pl.LightningModule):
    def __init__(self, model, processor, config):
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.predictions = []
        self.references = []

    def predict_step(self, batch, batch_idx):
        images = batch.pop("images")
        questions = batch.pop("questions")
        answers = batch.pop("answers")

        generated_answers = []
        for img, q in zip(images, questions):
            inputs = self.processor(images=img, text=q, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
            out_wo_prompt = outputs[:, inputs.input_ids.shape[-1]:]
            generated_text = self.processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
            generated_answers.append(generated_text)

        self.predictions.extend(generated_answers)
        self.references.extend(answers)

        return {"predictions": generated_answers, "references": answers}

    def on_predict_epoch_end(self):
        bleu_scores = [self._compute_bleu(ref, pred) for ref, pred in zip(self.references, self.predictions)]
        rouge_scores = [self.rouge_scorer.score(ref, pred) for ref, pred in zip(self.references, self.predictions)]

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_rouge1 = sum([s['rouge1'].fmeasure for s in rouge_scores]) / len(rouge_scores) if rouge_scores else 0
        avg_rouge2 = sum([s['rouge2'].fmeasure for s in rouge_scores]) / len(rouge_scores) if rouge_scores else 0
        avg_rougeL = sum([s['rougeL'].fmeasure for s in rouge_scores]) / len(rouge_scores) if rouge_scores else 0

        print(f"BLEU: {avg_bleu:.4f}")
        print(f"ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")

        self.predictions.clear()
        self.references.clear()

    def _compute_bleu(self, ground_truth, predicted_answer):
        reference = [ground_truth.split()]
        hypothesis = predicted_answer.split()
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
