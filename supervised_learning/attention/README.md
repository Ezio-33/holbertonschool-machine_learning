# 🧠 Transformer - Projet Attention

## 🎓 Projet de spécialisation - Holberton School

Ce projet consiste à construire, pas à pas, un **modèle Transformer complet** tel que présenté dans le célèbre article ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

L’objectif est d’implémenter les composants fondamentaux d’un **modèle de traduction automatique (seq2seq)** basé sur des mécanismes d’attention et sans RNN, 100% en TensorFlow.

---

## 📚 Ce que vous allez apprendre

✅ Créer un **RNN Encoder** et un **RNN Decoder** avec attention  
✅ Implémenter le **Self Attention** (scaled dot product attention)  
✅ Concevoir la **Multi-Head Attention**  
✅ Ajouter des **encodages positionnels**  
✅ Construire des **blocs Encoder / Decoder**  
✅ Assembler un modèle **Transformer complet** (Encodeur + Décodeur)  
✅ Comprendre les **masques** utilisés dans le NLP

---

## 🛠️ Technologies utilisées

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Normes [pycodestyle 2.11.1](https://pycodestyle.readthedocs.io/)
- Ubuntu 20.04 (WSL compatible)

---

## 📁 Structure du projet

| Fichier                       | Description                                      |
|------------------------------|--------------------------------------------------|
| `0-rnn_encoder.py`           | Encodeur RNN classique (GRU)                     |
| `1-self_attention.py`        | Mécanisme d’attention par produit scalaire      |
| `2-rnn_decoder.py`           | Décodeur RNN avec attention                     |
| `4-positional_encoding.py`   | Encodages de position sinusoïdaux               |
| `5-sdp_attention.py`         | Attention par produit scalaire mis à l’échelle  |
| `6-multihead_attention.py`   | Attention multi-tête                            |
| `7-transformer_encoder_block.py` | Bloc d’encodeur Transformer               |
| `8-transformer_decoder_block.py` | Bloc de décodeur Transformer               |
| `9-transformer_encoder.py`   | Encodeur complet (empilement de blocs)          |
| `10-transformer_decoder.py`  | Décodeur complet (empilement de blocs)          |
| `11-transformer.py`          | Modèle Transformer complet                      |
| `README.md`                  | Documentation du projet                         |

---


## ✅ Bonnes pratiques respectées

* 📎 Tous les fichiers respectent `pycodestyle`
* 📘 Chaque module, classe et fonction est documenté
* 🚫 Aucun import non autorisé
* 🧱 Code modulaire, réutilisable, structuré
