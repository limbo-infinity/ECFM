import torch
from pred_model import train_prediction_model, PricePredictor




if __name__ == "__main__":
    batch_size = 128
    obs_dim = 32
    hidden_dim = 256
    learning_rate = 1e-3
    num_epochs = 20
    T = 100
    K = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    g_phi = PricePredictor(obs_dim=obs_dim, hidden_dim=hidden_dim, T=T, K=K)
    g_phi.to(device)
    print(g_phi)

    optimizer = torch.optim.Adam(g_phi.parameters(), lr=learning_rate)

    num_train_samples = 1024
    xi_train = torch.randn(num_train_samples, obs_dim)
    da_train = torch.randn(num_train_samples, T, K)
    rt_train = torch.randn(num_train_samples, T, K)

   

    train_prediction_model(g_phi=g_phi, train_loader=train_loader, optimizer=optimizer,num_epochs=num_epochs,)
