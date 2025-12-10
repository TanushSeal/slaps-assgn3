import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

# DeepProbLog imports
from deepproblog.dataset import Dataset as DPL_Dataset
from deepproblog.dataset import DataLoader as DPL_DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.query import Query
from problog.logic import Term, Constant

# ==========================================
# PART 1: GLOBAL TENSOR STORE
# ==========================================
tensor_store = {}

# ==========================================
# PART 2: MODEL & LOGIC DEFINITION
# ==========================================

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10) 

    def forward_image(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def forward(self, x):
        indices = []
        # Robust Input Handling
        if isinstance(x, list):
            for item in x:
                if isinstance(item, list): item = item[0]
                if hasattr(item, 'value'): indices.append(int(item.value))
                else: indices.append(int(item))
        elif isinstance(x, torch.Tensor):
            indices = x.view(-1).tolist()
        else:
            if hasattr(x, 'value'): indices = [int(x.value)]
            else: indices = [int(x)]

        # Lookup
        device = next(self.parameters()).device
        images_list = [tensor_store[i] for i in indices]
        images_tensor = torch.stack(images_list).to(device)
        
        return self.forward_image(images_tensor)

# --- LOGIC RULES ---
logic_program_str = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

% 1. Sum
sum_digits(Idx1, Idx2, Sum) :- 
    digit(Idx1, N1), 
    digit(Idx2, N2), 
    Sum is N1 + N2.

% 2. Extension: Product
product_digits(Idx1, Idx2, Prod) :-
    digit(Idx1, N1),
    digit(Idx2, N2),
    Prod is N1 * N2.

% 3. Extension: Absolute Difference
diff_digits(Idx1, Idx2, Diff) :-
    digit(Idx1, N1),
    digit(Idx2, N2),
    Diff is abs(N1 - N2).
"""

# ==========================================
# PART 3: DATA & UTILS
# ==========================================

class ListDataset(DPL_Dataset):
    def __init__(self, queries):
        self.queries = queries
        
    def __len__(self): 
        return len(self.queries)
        
    def to_query(self, i): 
        return self.queries[i]
        
    def __getitem__(self, i):
        return self.queries[i]
        
    def __iter__(self):
        return iter(self.queries)

def create_dpl_dataset(mnist_data, op_name="sum_digits", limit=1000, offset=0):
    print(f"Creating {op_name} dataset with {limit} examples...")
    queries = []
    
    for i in range(limit):
        img1, l1 = mnist_data[2*i]
        img2, l2 = mnist_data[2*i+1]
        
        id1 = offset + (2*i)
        id2 = offset + (2*i + 1)
        tensor_store[id1] = img1
        tensor_store[id2] = img2
        
        # Safety: Ensure labels are pure Python ints
        l1 = int(l1)
        l2 = int(l2)
        
        if op_name == "sum_digits":
            res = l1 + l2
        elif op_name == "product_digits":
            res = l1 * l2
        elif op_name == "diff_digits":
            res = abs(l1 - l2)
            
        q = Query(Term(op_name, Constant(id1), Constant(id2), Constant(res)))
        queries.append(q)
        
    return ListDataset(queries)

def main():
    # 1. Setup Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)

    # Datasets
    train_dataset = create_dpl_dataset(mnist_train, "sum_digits", limit=1000, offset=0)
    
    test_sum = create_dpl_dataset(mnist_test, "sum_digits", limit=200, offset=10000)
    test_prod = create_dpl_dataset(mnist_test, "product_digits", limit=200, offset=20000)
    test_diff = create_dpl_dataset(mnist_test, "diff_digits", limit=200, offset=30000)

    # 2. Setup Network
    network = MNISTNet()
    net_obj = Network(network, "mnist_net", batching=True)
    net_obj.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # 3. Setup Model
    with open("model.pl", "w") as f:
        f.write(logic_program_str)
    
    model = Model("model.pl", [net_obj])
    model.set_engine(ExactEngine(model))
    
    # 4. Train
    print("\n--- Starting Neuro-Symbolic Training (2 Epochs) ---")
    train_model(model, DPL_DataLoader(train_dataset, batch_size=16), 25, log_iter=10)

    # 5. Evaluate
    def evaluate(dataset, name):
        print(f"\n--- Evaluating {name} Accuracy ---")
        correct = 0
        total = 0
        
        for i, query in enumerate(dataset):
            if query is None: continue
                
            # Solve
            res = model.solve([query])
            
            # --- FIX: Handle Dictionary Result ---
            if res:
                # The result is a dictionary {query: prob}
                # We extract the first (and only) value from this dictionary
                r_dict = res[0].result
                p = list(r_dict.values())[0]
                
                # Cast to float to avoid tensor errors
                if float(p) > 0.5:
                    correct += 1
            total += 1
            
        if total > 0:
            print(f"{name} Accuracy: {100 * correct / total:.2f}%")
        else:
            print(f"{name} Accuracy: N/A (No queries)")

    evaluate(test_sum, "Sum")
    evaluate(test_prod, "Product (Zero-Shot)")
    evaluate(test_diff, "Difference (Zero-Shot)")

    #Saving the learning

    print("\n--- Saving Model to 'mnist_neurosymbolic.pth' ---")
    torch.save(network.state_dict(), "mnist_neurosymbolic.pth")
    print("Model saved successfully!")

    #Cleanup

    if os.path.exists("model.pl"):
        os.remove("model.pl")

if __name__ == '__main__':
    main()