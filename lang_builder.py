import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# --- setup ---

concepts = ['dog', 'cat', 'car', 'tree', 'house']
tokens = ['a', 'b', 'c', 'd', 'e']

num_concepts = len(concepts)
num_tokens = len(tokens)

# --- models ---

class Sender(nn.Module):
    """
    The sender class takes in a concept and outputs a token.
    nn.Module is the base class for all neural network modules in PyTorch.
    Modules can contain other modules, allowing for nested structures. 
    They also have a forward method that defines the computation performed at every call.
    """

    def __init__(self):
        """
        The __init__ method initializes the Sender class. 
        It creates a linear layer that maps from the number of concepts to the number of tokens.
        """
        
        super().__init__()
        self.fc = nn.Linear(num_concepts, num_tokens) # we need this to map from the concept space to the token space
        # the purpose of mapping is to learn a representation of the concept in the token space, which can then be used to generate a token that represents the concept.
    
    def forward(self, concept_onehot):
        """
        The forward method defines the computation performed at every call.
        It takes in a concept one-hot vector, passes it through the linear layer, and returns the output.
        """

        logits = self.fc(concept_onehot) # the output of the linear layer is called logits, which are unnormalized scores for each token.

        probs = torch.softmax(logits, dim=-1) # we apply softmax to convert the logits into probabilities, 
        # which represent the likelihood of each token being the correct one for the given concept.
        
        return probs
    
class Receiver(nn.Module):
    """
    The receiver class takes in a token and outputs a concept.
    """

    def __init__(self):
        """
        The __init__ method initializes the Receiver class. 
        It creates a linear layer that maps from the number of tokens to the number of concepts.
        We do the same as the sender, but in reverse, because we want to map from the token space back to the concept space.
        """

        super().__init__()
        self.fc = nn.Linear(num_tokens, num_concepts) # notice how it goes tokens then concepts, which is the reverse of the sender.

    def forward(self, token):
        """
        The forward method defines the computation performed at every call.
        It takes in a token, passes it through the linear layer, and returns the output.
        """

        logits = self.fc(token)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
sender = Sender()
receiver = Receiver()

"""
Optimizers are used to update the parameters of the models during training.
We use the Adam optimizer, which is a popular optimization algorithm that combines the benefits of both AdaGrad and RMSProp.
lr is the learning rate, which controls how much we update the parameters of the model during training. 
A higher learning rate means more updates, while a lower learning rate means fewer updates.
"""
sender_optimizer = torch.optim.Adam(sender.parameters(), lr=0.01)
receiver_optimizer = torch.optim.Adam(receiver.parameters(), lr=0.01)
    
# --- training ---

for step in range(10000):
    # Pick random concept
    concept_index = random.randint(0, num_concepts - 1)

    # gets the one-hot encoding of the concept index, which is a vector of zeros with a 1 at the index of the concept.
    # one-hot encoding is a way to represent categorical variables as binary vectors, 
    # where each category is represented by a vector with a single 1 and the rest are 0s.
    concept_onehot = F.one_hot(torch.tensor(concept_index), num_concepts).float()

    # Pass the concept through the sender to get token probabilities, 
    # which is a vector of probabilities for each token being the correct one for the given concept.
    token_probs = sender(concept_onehot)

    # We create a categorical distribution from the token probabilities, 
    # which allows us to sample a token based on the probabilities output by the sender.
    token_dist = torch.distributions.Categorical(token_probs)

    # we sample a token index from the distribution, 
    # which gives us the index of the token that the sender has chosen to represent the concept.
    token_index = token_dist.sample()

    # we calculate the log probability of the sampled token index, 
    # which is used for the REINFORCE algorithm to update the sender's 
    # parameters based on the reward received from the receiver.
    log_prob = token_dist.log_prob(token_index)

    # we convert the token index into a one-hot vector, 
    # which is a vector of zeros with a 1 at the index of the token.
    token_onehot = F.one_hot(token_index, num_tokens).float()

    # we pass the token one-hot vector through the receiver to get logits, 
    # which are unnormalized scores for each concept.
    logits = receiver(token_onehot)

    # we calculate the cross-entropy loss between the logits and the true concept index, 
    # which measures how well the receiver is able to predict the correct concept based on the token provided by the sender.
    receiver_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([concept_index]))

    # predicted_concept is the index of the concept with the highest logit value.
    predicted_concept = torch.argmax(logits).item()

    # basic reward calculation
    reward = 1 if predicted_concept == concept_index else 0

    # ----- Update Receiver -----
    # we zero the gradients of the receiver optimizer, which clears any previously accumulated gradients from the receiver's parameters.
    receiver_optimizer.zero_grad()
    # we perform backpropagation on the receiver loss, which computes the gradients of the receiver's parameters with respect to the loss.
    # retain_graph=True is used to keep the computational graph in memory after backpropagation, 
    # which allows us to perform multiple backward passes on the same graph if needed.
    receiver_loss.backward(retain_graph=True)
    # we step the receiver optimizer, which updates the receiver's parameters based on the computed gradients.
    receiver_optimizer.step()

    # ----- Update Sender (REINFORCE) -----
    sender_loss = -log_prob * reward

    sender_optimizer.zero_grad()
    sender_loss.backward()
    sender_optimizer.step()

    # print results every 500 steps
    if step % 500 == 0:
        print(f"Step {step}: Concept: {concepts[concept_index]}, Token: {tokens[token_index]}, Predicted Concept: {concepts[predicted_concept]}, Reward: {reward}")

# print final mappings
print("\nFinal Mappings:")
for i in range(num_concepts):
    concept_onehot = F.one_hot(torch.tensor(i), num_concepts).float()
    token_probs = sender(concept_onehot)
    predicted_token_index = torch.argmax(token_probs).item()
    print(f"{concepts[i]} -> {tokens[predicted_token_index]}")