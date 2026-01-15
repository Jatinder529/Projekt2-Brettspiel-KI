import csv
import matplotlib.pyplot as plt
import os

def create_and_save_plot(variant_name, csv_path, output_filename):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping {variant_name}.")
        return

    epochs = []
    losses = []
    
    print(f"Processing {variant_name} from {csv_path}...")
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['epoch']))
                    losses.append(float(row['total_loss']))
                except ValueError:
                    continue 

        if not epochs:
            print(f"Warning: {csv_path} has no valid data.")
            return

        # Create a new figure for this plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, label='Total Loss', marker='o', color='b')
        
        plt.title(f"Training Loss: {variant_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        
        plt.savefig(output_filename)
        print(f"Saved graph to {output_filename}")
        plt.close() # Close to free memory/reset state

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

def main():
    # Variant 1
    create_and_save_plot("Variant 1 (5x5)", 
                         "./temp/variant1/loss_log.csv", 
                         "loss_variant_1.png")
    
    # Variant 2
    create_and_save_plot("Variant 2 (2 Priests)", 
                         "./temp/variant2/loss_log.csv", 
                         "loss_variant_2.png")
    
    # Tic-Tac-Toe
    create_and_save_plot("Tic-Tac-Toe", 
                         "./temp/tictactoe/loss_log.csv", 
                         "loss_tictactoe.png")
    
    print("Done! Check for loss_variant_1.png and loss_variant_2.png.")

if __name__ == "__main__":
    main()
