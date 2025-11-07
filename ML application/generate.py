import torch
import numpy as np
from model import CharLSTM
from dataset import Shakespeare
from main import one_hot_encode  # main.py에서 정의된 one_hot_encode 함수 사용

def generate(model, seed_characters, temperature, dataset, device, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        args: other arguments if needed

    Returns:
        samples: generated characters
    """

    model.eval()
    input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0)
    input_seq = one_hot_encode(input_seq, len(dataset.chars)).to(device)
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):
        hidden = tuple([h.to(device) for h in hidden])
    else:
        hidden = hidden.to(device)

    samples = seed_characters
    for _ in range(100):  # Generate 100 characters
        output, hidden = model(input_seq, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_char = torch.multinomial(output_dist, 1)[0]
        top_char = top_char % len(dataset.chars)
        char = dataset.idx_to_char[top_char.item()]
        samples += char
        input_seq = one_hot_encode(torch.tensor([[top_char]], dtype=torch.long), len(dataset.chars)).to(device)
    return samples

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Shakespeare('./hw3/data/shakespeare_train.txt')
    vocab_size = len(dataset.chars)
    hidden_size = 256  # Hidden size should match the size used during training
    n_layers = 2

    model = CharLSTM(vocab_size, hidden_size, vocab_size, n_layers).to(device)
    model.load_state_dict(torch.load('./hw3/model/lstm_model_best.pth', map_location=device))  # Load the trained model

    seed_chars_list = ['A', 'a', 'B', 'b', 'C']
    temperatures = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("Genearte text by changing seed chracters.")
    for seed_chars in seed_chars_list:
        generated_text = generate(model, seed_chars, 0.5, dataset, device)
        print("Seed chars: '{}', Temperature: 0.5".format(seed_chars))
        print(generated_text)
        print("- "*40)

    print('\n',"Genearte text by changing temperatures.")
    for temperature in temperatures:
            generated_text = generate(model, seed_chars_list[0], temperature, dataset, device)
            print("Seed chars: {}, Temperature: {}".format(seed_chars_list[0], temperature))
            print(generated_text)
            print("- "*40)
