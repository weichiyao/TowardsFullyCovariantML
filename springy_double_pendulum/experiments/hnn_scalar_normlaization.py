import os 
import jax.numpy as jnp
import jax.random as jr
jr.PRNGKey(42)
from model.scalaremlp import InvarianceLayer_objax 
from trainer.hamiltonian_dynamics import IntegratedDynamicsTrainer,DoubleSpringPendulum,hnnScalars_trial
from trainer.hamiltonian_dynamics import generate_trajectory_wz0s, GetHamiltonianDataset, GetHamiltonianDatasetWrapped
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo 
from oil.datasetup.datasets import split_dataset 
from oil.tuning.args import argupdated_config
import torch 
import logging 
import objax
import numpy as np




levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
          'warn': logging.WARNING,'warning': logging.WARNING,
          'info': logging.INFO,'debug': logging.DEBUG}

def compute_sqrt_trcov(xcov: jnp.ndarray):
    sqrt_trace = jnp.sqrt(jnp.trace(xcov) / 3)
    
    return sqrt_trace*jnp.eye(3), jnp.eye(3) / sqrt_trace

def compute_sqrt_cov(xcov: jnp.ndarray):
    """
    Compute the inverse square root of the sample covariance matrix of dataset X.

    Parameters:
    - xcov: A numpy array of shape (d, d) where d is the number of dimensions.

    Returns:
    - The inverse square root of the sample covariance matrix.
    """
    # Perform eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = jnp.linalg.eigh(xcov)
    if eigenvalues.min() < 0:
        print("The smallest eigen values is smaller than zero!")
    print("eigen values of the covariance matrix", eigenvalues.tolist())
    # Compute the diagonal matrix of the square root of eigenvalues
    lambda_sqrt     = jnp.diag(jnp.sqrt(eigenvalues))
    # Compute the diagonal matrix of the inverse square root of eigenvalues
    lambda_sqrt_inv = jnp.diag(1.0 / jnp.sqrt(eigenvalues))
    

    # Compute the square root of the covariance matrix
    cov_matrix_sqrt = jnp.dot(eigenvectors, jnp.dot(lambda_sqrt, eigenvectors.T))
    # Compute the inverse square root of the covariance matrix
    cov_matrix_inv_sqrt = jnp.dot(eigenvectors, jnp.dot(lambda_sqrt_inv, eigenvectors.T))
    if jnp.any(jnp.linalg.inv(cov_matrix_inv_sqrt) - cov_matrix_sqrt > 1e-6):
        print("something wrong with the covariance materix sqrt inverse!")
    return cov_matrix_sqrt, cov_matrix_inv_sqrt

def makeTrainerScalars(*,
                       dataset=DoubleSpringPendulum,
                       num_epochs=3000,  
                       ndata=1000,
                       seed=2021, 
                       bs=500,
                       lr=5e-3,
                       device='cuda',
                       split={'train':500,'val':.1,'test':.1},
                       data_config={'chunk_len':5,'dt':0.2,'integration_time':30,'regen':False,'root_dir':'/content/drive/MyDrive/Colab/DoublePendulum/Normalization/'},
                       net_config={'n_layers':3,'n_hidden':100}, 
                       log_level='info',
                       trainer_config={'log_dir':'/home/','log_args':{'minPeriod':.02,'timeFrac':.75},},
                       normalization_method='none',
                       do_OOD="test",
                       save=False,
                       trial=1):
    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    print("do OOD", do_OOD)
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(n_systems=ndata,**data_config)
        ## split_dataset_indices
        trajectories_dataset_filename = base_ds.filename.split('/')[-1]  
        split_indices_filename = os.path.join(
            base_ds.root_dir, 
            '_'.join(['splitindices_' + '_'.join([str(x) for x in list(split.values())]), 
                    trajectories_dataset_filename]))
        if os.path.exists(split_indices_filename) and not data_config['regen']:
            print(f"Loading split_indices from {split_indices_filename}...")
            split_indices = torch.load(split_indices_filename)
        else:
            print(f"File {split_indices_filename} does not exist or regen=True. Creating the split_indices ...")
            datasets = split_dataset(base_ds,splits=split)
            split_indices = {'train': datasets['train']._ids, 'test': datasets['test']._ids, 'val': datasets['val']._ids} 
            
            torch.save(split_indices, split_indices_filename)
            print(f"Saved file {split_indices_filename}...")

        datasets = {}
        for split_name in ['train', 'test', 'val']: 
            if split_name == 'test':
                print(np.asarray(base_ds.Zs_long[split_indices[split_name]]).shape)
                datasets[split_name] = GetHamiltonianDataset(np.asarray(base_ds.Zs_long[split_indices[split_name]]), np.asarray(base_ds.T_long))
            else:
                datasets[split_name] = GetHamiltonianDataset(base_ds.Zs[split_indices[split_name]], base_ds.T)
        
    if do_OOD != "none":  
        OOD_trajectories_dataset_filename = os.path.join(
            base_ds.root_dir, 
            f'{do_OOD}OOD_'+split_indices_filename.split('/')[-1])
        if os.path.exists(OOD_trajectories_dataset_filename) and not data_config['regen']:
            print(f"Loading {do_OOD} OOD trajectories from {OOD_trajectories_dataset_filename}...")
            zs_all = torch.load(OOD_trajectories_dataset_filename)
        else:
            print(f"File {OOD_trajectories_dataset_filename} does not exist or regen=True. Creating the {do_OOD} OOD trajectories ...")
            zs_all = generate_trajectory_wz0s(base_ds.H, 
                                            base_ds.Zs[split_indices[do_OOD],0,:], 
                                            ts=datasets['test'].ts, 
                                            bs=512)
            torch.save(zs_all, OOD_trajectories_dataset_filename)
            print(f"Saved file {OOD_trajectories_dataset_filename}...")
        datasets['test'].zs = zs_all
    
    #########################################################################
    normalization = {'method': normalization_method}
    if normalization['method'] == "none":
        pass
    if normalization['method'] == "covariant":
        ## Set normalizing statistics
        normalization['xmean_q'] = jnp.zeros(3,)
        normalization['xmean_p'] = jnp.zeros(3,)
          
        ## Compute normalizing statistics w.r.t. initial time points in training dataset 
        z0 = datasets['train'].zs[:,0,:].reshape(-1,4,3)
        n = datasets['train'].zs.shape[0] * 2
          
        Qtilde = z0[:,:2,:].reshape(-1,3)   # (3,3)
        Ptilde = z0[:,2:,:].reshape(-1,3)   # (3,3)
        xcov_q = jnp.transpose(Qtilde) @ Qtilde / n          # (3,3)
        xcov_p = jnp.transpose(Ptilde) @ Ptilde / n          # (3,3)
            
        normalization['xstd_q'], normalization['xstd_inv_q'] = compute_sqrt_trcov(xcov_q)
        normalization['xstd_p'], normalization['xstd_inv_p'] = compute_sqrt_trcov(xcov_p) 

        ## Normalize the whole training data set (both input and output) w.r.t. normalizing statistics
        zs = datasets['train'].zs.reshape(*datasets['train'].zs.shape[:-1],4,3)
        zs_q=(zs[...,:2,:] - normalization['xmean_q']) @ normalization['xstd_inv_q']
        zs_p=(zs[...,2:,:] - normalization['xmean_p']) @ normalization['xstd_inv_p']
        zs_normalized = jnp.concatenate([zs_q,zs_p],axis=-2).reshape(*datasets['train'].zs.shape[:-1],12) # (n, 12)

        datasets['train'].zs = np.asarray(zs_normalized)
    elif normalization['method'] == "colwise":
        z0 = datasets['train'].zs[:,0,:] 
        ## Set normalizing statistics
        normalization['xmean'] = jnp.mean(z0, axis=0)        # (12,)
        normalization['xstd']  = jnp.std(z0, axis=0)         # (12,)

        zs_normalized = (datasets['train'].zs-normalization['xmean']) / normalization['xstd']  # (n, 12)

        datasets['train'].zs = np.asarray(zs_normalized)
    else:
        print(f"Received normalization_method={normalization_method}. Only allowed 'covariant', 'colwise', 'none'.")
    #########################################################################
    
    dataloaders = {k:LoaderTo(DataLoader(GetHamiltonianDatasetWrapped(v),
                                         batch_size=min(bs,len(v)),
                                         shuffle=(k=='train'),
                   num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    
    model = InvarianceLayer_objax(**net_config)
    opt_constr = objax.optimizer.Adam    
    lr_sched = lambda e: lr if (e < 200) else (lr*0.6 if e < 1500 else (lr*0.4 if e<2500 else lr*0.2)) 
    return IntegratedDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,normalization,**trainer_config)

if __name__ == "__main__":
    Trial = hnnScalars_trial(makeTrainerScalars)
    cfg,outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))
    print(outcome)