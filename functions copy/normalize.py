import torch

def normalize_lab(lab_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(lab_img.size())
    stds = torch.zeros(lab_img.size())
    
    mean[:,0,:,:] = 50
    mean[:,1,:,:] = 0
    mean[:,2,:,:] = 0
    
    stds[:,0,:,:] = 50
    stds[:,1,:,:] = 128
    stds[:,2,:,:] = 128
    
    return (lab_img.double() - mean.double())/stds.double()

def normalize_seg(seg):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    result = seg[:,0,:,:]
    if torch.max(result) >1:
        result = result/100.0
    result = torch.round(result)
    
    
    return result

def normalize_rgb(rgb_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(rgb_img.size())
    stds = torch.zeros(rgb_img.size())
    
    mean[:,0,:,:] = 0.485
    mean[:,1,:,:] = 0.456
    mean[:,2,:,:] = 0.406
    
    stds[:,0,:,:] = 0.229
    stds[:,1,:,:] = 0.224
    stds[:,2,:,:] = 0.225
    
    return (rgb_img.double() - mean.double())/stds.double()
   
    
def denormalize_lab(lab_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(lab_img.size())
    stds = torch.zeros(lab_img.size())
    
    mean[:,0,:,:] = 50
    mean[:,1,:,:] = 0
    mean[:,2,:,:] = 0
    
    stds[:,0,:,:] = 50
    stds[:,1,:,:] = 128
    stds[:,2,:,:] = 128

    return lab_img.double() *stds.double() + mean.double()


def denormalize_rgb(rgb_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(rgb_img.size())
    stds = torch.zeros(rgb_img.size())
    
    mean[:,0,:,:] = 0.485
    mean[:,1,:,:] = 0.456
    mean[:,2,:,:] = 0.406
    
    stds[:,0,:,:] = 0.229
    stds[:,1,:,:] = 0.224
    stds[:,2,:,:] = 0.225

    return rgb_img.double() *stds.double() + mean.double()