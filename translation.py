from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class translator():
    def __init__(self):
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    def to_eng(self, text):
        if self.model and self.tokenizer:
            self.tokenizer.src_lang = "ar_AR"

            encoded_ar = self.tokenizer(text, return_tensors="pt")
            generated_tokens = self.model.generate(
                **encoded_ar,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"]
            )
            eng_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            return eng_text
        return text