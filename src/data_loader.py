"""Load pickle data from optimization folders"""
import os
import glob
import pickle
import numpy as np
import torch

def load_pickle_data(data_dir, latent_dim=30):
    """
    Load latent vectors and metadata from pickle files.
    
    Args:
        data_dir: Directory containing .pkl files
        latent_dim: Expected latent dimension (default: 30)
    
    Returns:
        latents: numpy array of shape [N, latent_dim]
        rmsd: numpy array of shape [N]
        energy: numpy array of shape [N]
        gen: numpy array of shape [N]
    """
    data_dir = os.path.abspath(data_dir)
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    
    z_list, E_list, lr_list, gen_list = [], [], [], []
    
    for file in pkl_files:
        try:
            with open(file, 'rb') as f:
                ld = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
            continue
        
        # Check for required keys
        if 'scores' not in ld or 'lig_rmsd' not in ld['scores'] or 'z' not in ld:
            continue
        
        z_list.append(ld['z'])
        E_list.append(ld['scores'].get('energy', 0.0))
        lr_list.append(ld['scores']['lig_rmsd'])
        gen_list.append(ld.get('curr_gen', 0))
    
    if len(z_list) == 0:
        return np.array([]).reshape(0, latent_dim), np.array([]), np.array([]), np.array([])
    
    # Stack latent vectors
    if isinstance(z_list[0], torch.Tensor):
        latents = torch.stack(z_list).cpu().numpy()
    else:
        latents = np.stack([np.array(z) for z in z_list])
    
    # Ensure latents are 2D
    if latents.ndim == 1:
        latents = latents.reshape(1, -1)
    elif latents.ndim > 2:
        latents = latents.reshape(len(z_list), -1)
    
    return latents, np.array(lr_list), np.array(E_list), np.array(gen_list)