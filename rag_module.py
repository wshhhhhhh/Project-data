import json
import os

class DR_RAG:

    def __init__(self, path="guidelines.json"):

        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, path)

        with open(path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

    def retrieve(self, label):

        mapping = {
            0: "normal",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "proliferative"
        }

        key = mapping.get(label, "mild")

        return [
            d["text"] for d in self.docs
            if key in d["disease"]
        ] or [self.docs[0]["text"]]

    def generate_report(self, label, confidence):

        texts = self.retrieve(label)

        return {
            "prediction": label,
            "confidence": confidence,
            "docs": texts,
            "template_report": "\n\n".join(texts)
        }