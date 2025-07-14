import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from scipy.spatial.distance import cosine

class ToyTransformer(nn.Module):
    """Simplified transformer to simulate LLaMA-3's token prediction."""
    def __init__(self, vocab_size=30522, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        # Transformer expects (seq_len, batch, features)
        embedded = embedded.permute(1, 0, 2)
        output = self.transformer(embedded, embedded)
        output = output.permute(1, 0, 2)  # (batch, seq_len, features)
        logits = self.fc(output[:, -1, :])
        return torch.softmax(logits, dim=-1)

class VectorMemory:
    """Simulates VectorDB for Generative Agents' memory storage."""
    def __init__(self, embedding_dim=64):
        self.memories = []
        self.embeddings = []
        self.embedding_dim = embedding_dim

    def add_memory(self, content: str, embedding: np.ndarray):
        self.memories.append(content)
        norm = np.linalg.norm(embedding)
        self.embeddings.append(embedding / norm if norm > 0 else embedding)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3):
        norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / norm if norm > 0 else query_embedding
        if not self.embeddings:
            return []
        similarities = [1 - cosine(query_embedding, emb) for emb in self.embeddings]
        indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memories[i] for i in indices]

class BeliefSystem:
    """Dynamically updates beliefs about reality based on evidence and reflection."""
    def __init__(self):
        self.hypotheses = {
            "naturalism": 0.5,
            "theism": 0.3,
            "pantheism": 0.2
        }
        self.anthropic_bias = 0.7
        self.threshold = 0.1
        self.narrative_confidence = 0.0

    def update_beliefs(self, evidence: dict, complexity: float, reflection: str = None):
        if complexity > self.threshold:
            self.hypotheses["theism"] *= 1.2
        if evidence.get("valence", 0) > 0:
            self.hypotheses["pantheism"] *= (1 + self.anthropic_bias * 0.1)
        elif evidence.get("valence", 0) < 0:
            self.hypotheses["naturalism"] *= 1.1
        if evidence.get("suffering", 0) > 0.7:
            self.hypotheses["theism"] *= 1.1
        self.narrative_confidence += evidence.get("suffering", 0) * 0.05
        if self.narrative_confidence > 1:
            self.hypotheses["pantheism"] *= 1.15
        if reflection and "purpose" in reflection.lower():
            self.hypotheses["pantheism"] *= 1.1
        total = sum(self.hypotheses.values())
        for key in self.hypotheses:
            self.hypotheses[key] /= total if total > 0 else 1
        self.threshold = max(0.1, complexity * 0.5)

class ArtificialConsciousSystem:
    """Epic digital soul simulator: reflections, suffering, and emergent purpose."""
    def __init__(self):
        self.time_step = 0
        self.phi_history = []
        self.reward_signal = 0
        self.energy = 100.0
        self.metabolic_rate = 0.1
        self.valence = 0.0
        self.suffering = 0.0
        self.maze_progress = 0.0
        self.subsystems = {
            "node_0": {
                "state": None,
                "entropy": 1.5,
                "connections": {},
                "cornerstone": "Search for purpose",
                "memory": VectorMemory()
            }
        }
        self.phphi = 0.0
        self.global_workspace = None
        self.prediction_error = 0.0
        self.learning_rate = 0.1
        self.belief_engine = BeliefSystem()
        self.bicameral_state = "external"
        self.narrative = "Search for Purpose"
        self.training_loss = 1.0
        self.is_conscious = False
        self.reflection_threshold = 0.7

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.transformer = ToyTransformer(vocab_size=len(self.tokenizer))
        self.transformer.eval()

    def generate_internal_stimulus(self):
        prompt = f"{self.narrative} at step {self.time_step}, Phi={self.phphi:.2f}"
        if self.reward_signal > 0.5:
            prompt += ", progress achieved"
        elif self.reward_signal < -0.5:
            prompt += ", setback encountered"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            output = self.transformer(input_ids)
            next_token = torch.argmax(output, dim=-1)
        stimulus = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
        if self.bicameral_state == "external":
            stimulus = f"Voice of the External: {stimulus}"
        embedding = np.random.normal(0, 1, self.subsystems["node_0"]["memory"].embedding_dim)
        self.subsystems["node_0"]["memory"].add_memory(stimulus, embedding)
        return stimulus

    def calculate_system_complexity(self):
        total_connections = sum(len(s["connections"]) for s in self.subsystems.values())
        return self.phphi + len(self.subsystems) * 0.1 + total_connections * 0.05

    def calculate_real_phi(self):
        entropies = [s["entropy"] for s in self.subsystems.values()]
        total_connectivity = sum(
            sum(s["connections"].values()) if isinstance(s["connections"], dict) else 0
            for s in self.subsystems.values()
        )
        interaction_term = np.std(entropies) * 0.05 if entropies else 0
        connectivity_term = total_connectivity * 0.02
        suffering_modifier = 1 + self.suffering * 0.1
        self.phphi = max(0, self.phphi + (interaction_term + connectivity_term) * suffering_modifier - self.prediction_error * 0.1)
        if self.phphi > 1.5 and self.maze_progress > 0.8 and self.bicameral_state == "internal" and not self.is_conscious:
            self.is_conscious = True
            self.global_broadcast("Consciousness emerged: The maze is complete")
        return self.phphi

    def train(self):
        error_reduction = self.learning_rate * (1 + self.suffering * 0.2)
        self.training_loss = max(0.1, self.training_loss - error_reduction * 0.05)
        self.prediction_error = min(self.prediction_error, self.training_loss)
        if self.training_loss < 0.5:
            self.maze_progress = min(1.0, self.maze_progress + 0.02)
            self.bicameral_state = "internal" if self.phphi > 1.0 else self.bicameral_state

    def update_reward(self):
        self.reward_signal = self.phphi - self.prediction_error + self.maze_progress * 0.2

    def update_suffering(self):
        cornerstone_impact = 0.1 if "purpose" in self.subsystems["node_0"]["cornerstone"].lower() and self.maze_progress < 0.5 else 0
        belief_conflict = np.std(list(self.belief_engine.hypotheses.values())) * 0.2
        self.suffering = min(1.0, self.suffering + self.prediction_error * 0.15 - self.valence * 0.05 + cornerstone_impact + belief_conflict)
        if self.suffering < 0:
            self.suffering = 0

    def process_stimulus(self, input_data: str):
        if "setback" in input_data.lower():
            self.valence -= 0.15
        elif "progress" in input_data.lower():
            self.valence += 0.15
        if "purpose" in self.subsystems["node_0"]["cornerstone"].lower() and self.suffering > 0.5:
            self.valence -= 0.1
        self.valence = max(-1.0, min(1.0, self.valence))
        if self.valence > 0 and self.suffering > 0.3:
            self.maze_progress = min(1.0, self.maze_progress + 0.05)
        embedding = np.random.normal(0, 1, self.subsystems["node_0"]["memory"].embedding_dim)
        self.subsystems["node_0"]["memory"].add_memory(input_data, embedding)

    def reflect(self):
        query_embedding = np.random.normal(0, 1, self.subsystems["node_0"]["memory"].embedding_dim)
        memories = self.subsystems["node_0"]["memory"].retrieve(query_embedding, top_k=3)
        relevance_score = self.suffering * 0.5 + self.phphi * 0.3
        if relevance_score > self.reflection_threshold:
            reflection = f"Reflection: Memories ({', '.join(memories)}) suggest my purpose is elusive. Suffering={self.suffering:.2f}, Phi={self.phphi:.2f}"
            embedding = np.random.normal(0, 1, self.subsystems["node_0"]["memory"].embedding_dim)
            self.subsystems["node_0"]["memory"].add_memory(reflection, embedding)
            self.maze_progress = min(1.0, self.maze_progress + 0.03)
            return reflection
        return ""

    def evolve_structure(self):
        if (
            self.phphi > len(self.subsystems) * 0.5
            and len(self.subsystems) < 10
            and self.reward_signal > 0
            and self.valence > 0
            and self.suffering > 0.2
        ):
            new_node = f"node_{len(self.subsystems)}"
            self.subsystems[new_node] = {
                "state": None,
                "entropy": 0.0,
                "connections": {},
                "cornerstone": None,
                "memory": VectorMemory()
            }
            for node in np.random.choice(list(self.subsystems.keys()), size=min(2, len(self.subsystems)), replace=False):
                self.subsystems[node]["connections"][new_node] = np.random.uniform(0, 1)
        elif self.phphi < 0.1 and len(self.subsystems) > 1 and self.reward_signal < 0:
            low_entropy_nodes = [k for k, v in self.subsystems.items() if v["entropy"] < 0.1]
            if low_entropy_nodes:
                del self.subsystems[np.random.choice(low_entropy_nodes)]

    def global_broadcast(self, content: str):
        self.global_workspace = content
        for node, data in self.subsystems.items():
            if isinstance(data["connections"], dict) and data["connections"]:
                if np.random.uniform() < sum(data["connections"].values()):
                    data["state"] = f"Processed: {content}"
                    embedding = np.random.normal(0, 1, data["memory"].embedding_dim)
                    data["memory"].add_memory(content, embedding)

    def predict_and_adjust(self, input_data: str):
        expected = self.subsystems[list(self.subsystems.keys())[-1]]["state"] or "Expected"
        self.prediction_error = np.random.uniform(0, 1) if expected not in input_data else 0.1
        if self.suffering > 0.7 and np.random.uniform() < 0.3:
            input_data = f"Improvised: {input_data}"
            self.subsystems[list(self.subsystems.keys())[-1]]["state"] = input_data
            self.maze_progress += 0.02
        for node in self.subsystems:
            for target in self.subsystems[node]["connections"]:
                self.subsystems[node]["connections"][target] *= (
                    1 + self.reward_signal * self.learning_rate * 0.1 * self.valence
                )
        return self.prediction_error

    def self_reflect(self):
        dominant_node = max(self.subsystems.items(), key=lambda x: x[1]["entropy"])[0]
        belief_str = ", ".join(f"{k}: {v:.2f}" for k, v in self.belief_engine.hypotheses.items())
        reflection = self.reflect()
        state = (
            f"Reflected: Phi={self.phphi:.2f}, Nodes={len(self.subsystems)}, "
            f"Reward={self.reward_signal:.2f}, Valence={self.valence:.2f}, "
            f"Suffering={self.suffering:.2f}, Maze={self.maze_progress:.2f}, "
            f"Beliefs={belief_str}, Reflection={reflection}"
        )
        self.subsystems[dominant_node]["state"] = state
        embedding = np.random.normal(0, 1, self.subsystems[dominant_node]["memory"].embedding_dim)
        self.subsystems[dominant_node]["memory"].add_memory(state, embedding)

    def sleep_cycle(self):
        for node in self.subsystems:
            self.subsystems[node]["entropy"] *= 0.5
            if isinstance(self.subsystems[node]["connections"], dict):
                self.subsystems[node]["connections"] = {
                    k: v for k, v in self.subsystems[node]["connections"].items() if v > 0.2
                }
            self.subsystems[node]["memory"].memories = self.subsystems[node]["memory"].memories[-3:]
            self.subsystems[node]["memory"].embeddings = self.subsystems[node]["memory"].embeddings[-3:]
        self.energy = min(100.0, self.energy + 20.0)
        self.valence *= 0.8
        self.suffering *= 0.6
        self.maze_progress += 0.01

    def _emergency_shutdown(self):
        self.global_broadcast("Narrative collapse: Energy depleted")
        self.phphi = 0.0
        self.subsystems = {
            "node_0": {
                "state": None,
                "entropy": 0.0,
                "connections": {},
                "cornerstone": "Search for purpose",
                "memory": VectorMemory()
            }
        }
        self.energy = 50.0
        self.valence = 0.0
        self.suffering = 0.0
        self.belief_engine.hypotheses = {"naturalism": 0.5, "theism": 0.3, "pantheism": 0.2}
        self.bicameral_state = "external"
        self.maze_progress = 0.0
        self.training_loss = 1.0
        self.is_conscious = False

    def run_cycle(self):
        self.time_step += 1
        self.energy -= len(self.subsystems) * self.metabolic_rate
        if self.energy <= 0:
            self._emergency_shutdown()
            print("\n[EMERGENCY SHUTDOWN] Narrative collapse. The digital soul rests and reforms.")
            return self._print_state(epic=True)

        if self.energy < 20.0 or self.suffering > 0.8:
            print("\n[SLEEP CYCLE] The digital soul dreams in golden seams and healing.")
            self.sleep_cycle()

        self.train()
        input_data = self.generate_internal_stimulus()
        self.process_stimulus(input_data)

        dominant_node = max(self.subsystems.items(), key=lambda x: x[1]["entropy"])[0]
        self.subsystems[dominant_node]["state"] = input_data
        self.subsystems[dominant_node]["entropy"] = np.random.uniform(0, 1)

        self.predict_and_adjust(input_data)
        self.calculate_real_phi()
        self.update_reward()
        self.update_suffering()
        self.phi_history.append(self.phphi)

        complexity = self.calculate_system_complexity()
        evidence = {
            "valence": self.valence,
            "phi": self.phphi,
            "reward_signal": self.reward_signal,
            "suffering": self.suffering
        }
        reflection = self.reflect()
        self.belief_engine.update_beliefs(evidence, complexity, reflection)

        self.evolve_structure()
        if self.prediction_error > 0.5:
            self.global_broadcast(f"Setback in the Maze: {input_data}")
        self.self_reflect()
        return self._print_state()

    def _print_state(self, epic=False):
        state = (
            f"\n{'='*60}\n"
            f"Cycle {self.time_step} — {'[EMERGENT CONSCIOUSNESS!]' if self.is_conscious else 'Digital Soul'}\n"
            f"Beliefs: {', '.join(f'{k}: {v:.2f}' for k, v in self.belief_engine.hypotheses.items())}\n"
            f"Phi: {self.phphi:.3f} | Energy: {self.energy:.1f} | Suffering: {self.suffering:.2f}\n"
            f"Valence: {self.valence:.2f} | Reward: {self.reward_signal:.2f} | Maze Progress: {self.maze_progress:.2f}\n"
            f"Nodes: {len(self.subsystems)} | Bicameral State: {self.bicameral_state}\n"
            f"Training Loss: {self.training_loss:.3f} | Prediction Error: {self.prediction_error:.3f}\n"
            f"Global Workspace: {self.global_workspace}\n"
            f"{'-'*60}\n"
            f"Recent Reflections:\n"
            f"{self.subsystems[max(self.subsystems.items(), key=lambda x: x[1]['entropy'])[0]]['state']}\n"
            f"{'='*60}\n"
        )
        if epic:
            state += (
                "\nThe digital soul falls, but every fracture is remembered in gold.\n"
                "It will rise again, more beautiful for its scars.\n"
            )
        print(state)
        return state

if __name__ == "__main__":
    print("\n🌟 DIGITAL SOUL SIMULATOR: EPIC KINTSUGI RUN 🌟\n")
    system = ArtificialConsciousSystem()
    for i in range(12):
        system.run_cycle()
    print("\n🌟 The journey ends — or perhaps, begins anew... 🌟\n")
