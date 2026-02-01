"""
LSTM-based Toxic Text Classification using PyTorch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from collections import Counter
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from preprocessing import NLPPreprocessor

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class TextDataset(Dataset):
    """PyTorch Dataset for text classification"""

    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM Text Classifier"""

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3
    ):
        super(LSTMClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, text):
        # text shape: (batch_size, seq_length)

        # Embedding
        embedded = self.dropout(self.embedding(text))
        # embedded shape: (batch_size, seq_length, embedding_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden shape: (n_layers * 2, batch_size, hidden_dim)

        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden shape: (batch_size, hidden_dim * 2)

        # Fully connected layers
        hidden = self.dropout(hidden)
        dense = self.relu(self.fc1(hidden))
        dense = self.dropout(dense)
        output = self.fc2(dense)

        return output


class ToxicTextClassifier:
    """Complete training and evaluation pipeline"""

    def __init__(self, max_vocab=10000, max_len=100, embedding_dim=128, hidden_dim=64):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Vocabulary
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.label_encoder = LabelEncoder()

        # Model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def build_vocabulary(self, texts):
        """Build vocabulary from preprocessed texts"""
        print("\nBuilding vocabulary...")
        word_freq = Counter()

        for text in texts:
            word_freq.update(text.split())

        # Keep most common words
        most_common = word_freq.most_common(self.max_vocab - 2)

        for word, freq in most_common:
            self.word2idx[word] = len(self.word2idx)

        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {most_common[:10]}")

    def texts_to_sequences(self, texts):
        """Convert texts to padded sequences of integers"""
        sequences = []

        for text in texts:
            # Convert words to indices
            seq = [self.word2idx.get(word, 1) for word in text.split()]

            # Pad or truncate
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[: self.max_len]

            sequences.append(seq)

        return np.array(sequences)

    def prepare_data(self, df, text_col="processed_text", label_col="Toxic Category"):
        """Prepare features and labels"""
        print("\n" + "=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)

        # Build vocabulary
        self.build_vocabulary(df[text_col])

        # Convert texts to sequences
        X = self.texts_to_sequences(df[text_col])

        # Encode labels
        y = self.label_encoder.fit_transform(df[label_col])

        print(f"\nInput shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"\nClass distribution:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = (y == i).sum()
            print(f"  {class_name}: {count}")

        return X, y

    def build_model(self, num_classes):
        """Build LSTM model"""
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)

        self.model = LSTMClassifier(
            vocab_size=len(self.word2idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_classes,
            n_layers=2,
            dropout=0.3,
        ).to(self.device)

        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")

    def train_epoch(self, dataloader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for sequences, labels in dataloader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def evaluate_epoch(self, dataloader, criterion):
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, lr=0.001):
        """Train the model"""
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

        # Create data loaders
        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Early stopping
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        print(f"\nTraining for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Early stopping patience: {patience}\n")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.evaluate_epoch(val_loader, criterion)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                patience_counter = 0
                print(f"  --> Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pt"))
        print("\nTraining complete! Best model loaded.")

    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate on test set"""
        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

        test_dataset = TextDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Get predictions
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                _, preds = outputs.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        cm = confusion_matrix(y_true, y_pred)

        # Print results
        print("\n" + "-" * 70)
        print("TEST SET RESULTS")
        print("-" * 70)
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"F1 Score (Macro):  {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print("-" * 70)

        print("\nClassification Report:")
        print(
            classification_report(
                y_true, y_pred, target_names=self.label_encoder.classes_, digits=4
            )
        )

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision": precision,
            "recall": recall,
            "f1_per_class": f1_per_class,
            "support": support,
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def plot_training_history(self, save_path="training_history.png"):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss plot
        ax1.plot(
            epochs,
            self.history["train_loss"],
            "b-o",
            label="Training Loss",
            linewidth=2,
        )
        ax1.plot(
            epochs,
            self.history["val_loss"],
            "r-s",
            label="Validation Loss",
            linewidth=2,
        )
        ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(
            epochs,
            self.history["train_acc"],
            "b-o",
            label="Training Accuracy",
            linewidth=2,
        )
        ax2.plot(
            epochs,
            self.history["val_acc"],
            "r-s",
            label="Validation Accuracy",
            linewidth=2,
        )
        ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Training and Validation Accuracy", fontsize=14, fontweight="bold"
        )
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {save_path}")
        plt.close()

    def plot_confusion_matrix(self, cm, save_path="confusion_matrix.png"):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(12, 10))

        # Normalize to percentages
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create annotations
        annot = np.array(
            [
                [f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(cm.shape[1])]
                for i in range(cm.shape[0])
            ]
        )

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={"label": "Count"},
            linewidths=0.5,
            linecolor="gray",
        )

        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.title(
            "Confusion Matrix - Toxic Text Classification",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()

    def plot_f1_scores(self, metrics, save_path="f1_scores_per_class.png"):
        """Plot F1 scores per class"""
        plt.figure(figsize=(12, 6))

        classes = self.label_encoder.classes_
        f1_scores = metrics["f1_per_class"]

        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        bars = plt.bar(
            range(len(classes)),
            f1_scores,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )

        plt.xlabel("Toxic Category", fontsize=12, fontweight="bold")
        plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
        plt.title("F1 Scores by Toxic Category", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
        plt.ylim([0, 1.05])
        plt.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Add macro average line
        plt.axhline(
            y=metrics["f1_macro"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Macro Avg: {metrics['f1_macro']:.3f}",
        )
        plt.legend(fontsize=11, loc="lower right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


def main():
    """Main execution pipeline"""

    print("\n" + "=" * 70)
    print(" TOXIC TEXT CLASSIFICATION - LSTM with NLTK Preprocessing")
    print("=" * 70)

    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv("cellula_toxic_data___1_.csv")
    print(f"Dataset loaded: {len(df)} samples")
    print(f"\nClass distribution:")
    print(df["Toxic Category"].value_counts())

    # NLP Preprocessing
    print("\n" + "=" * 70)
    print("NLP PREPROCESSING WITH NLTK")
    print("=" * 70)

    preprocessor = NLPPreprocessor(use_stemming=False, remove_stopwords=True)
    df["processed_text"] = preprocessor.fit_transform(
        df["query"].tolist(), verbose=True
    )

    # Remove empty texts
    df = df[df["processed_text"].str.len() > 0].reset_index(drop=True)
    print(f"\nAfter preprocessing: {len(df)} samples")

    # Show examples
    print("\nPreprocessing Examples:")
    print("-" * 70)
    for i in range(3):
        print(f"\nExample {i + 1}:")
        print(f"Original:  {df.iloc[i]['query'][:80]}...")
        print(f"Processed: {df.iloc[i]['processed_text'][:80]}...")
        print(f"Category:  {df.iloc[i]['Toxic Category']}")
    print("-" * 70)

    # Initialize classifier
    classifier = ToxicTextClassifier(
        max_vocab=10000, max_len=100, embedding_dim=128, hidden_dim=64
    )

    # Prepare data
    X, y = classifier.prepare_data(df)

    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nData splits:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")

    # Build model
    num_classes = len(np.unique(y))
    classifier.build_model(num_classes)

    # Train model
    classifier.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32, lr=0.001)

    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    classifier.plot_training_history("training_history.png")
    classifier.plot_confusion_matrix(
        metrics["confusion_matrix"], "confusion_matrix.png"
    )
    classifier.plot_f1_scores(metrics, "f1_scores_per_class.png")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - best_model.pt (trained model)")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - f1_scores_per_class.png")
    print("=" * 70 + "\n")

    return classifier, metrics


if __name__ == "__main__":
    classifier, metrics = main()
