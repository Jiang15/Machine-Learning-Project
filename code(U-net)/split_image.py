#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
from skimage import io


# In[40]:


im = io.imread('./data/frame/Denis_Joly_ADH1_27.1_frames.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/2_DJA_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[41]:


im = io.imread('./data/image/mask/Denis_Joly_ADH1_27.1_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/2_DJA_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[42]:


im = io.imread('./data/image/frame/Denis_Joly_V10(10)_1.1_frames.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/2_DJV1_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[45]:


im = io.imread('./data/image/mask/Denis_Joly_V10(10)_1.1_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/2_DJV1_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[46]:


im = io.imread('./data/image/frame/Denis_Joly_V10(10)_2.1_frames.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/2_DJV2_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[47]:


im = io.imread('./data/image/mask/Denis_Joly_V10(10)_2.1_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/2_DJV2_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[48]:


im = io.imread('./data/image/frame/augoustina_first_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_A1_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[49]:


im = io.imread('./data/image/mask/augoustina_first_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_A1_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage][500:][:-500])


# In[50]:


im = io.imread('./data/image/frame/augoustina_second_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_A2_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[51]:


im = io.imread('./data/image/mask/augoustina_second_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_A2_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[52]:


im = io.imread('./data/image/frame/michael_1.2_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_M12_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[53]:


im = io.imread('./data/image/mask/michael_1.2_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_M12_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[54]:


im = io.imread('./data/image/frame/michael_1.2.2_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_M122_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[55]:


im = io.imread('./data/image/mask/michael_1.2.2_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_M122_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[56]:


im = io.imread('./data/image/frame/michael_4.1_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_M41_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[57]:


im = io.imread('./data/image/mask/michael_4.1_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_M41_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])

im = io.imread('./data/image/frame/denis_ADH1_1.1_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_DA11_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[57]:


im = io.imread('./data/image/mask/denis_ADH1_1.1_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_DA11_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])

# In[ ]:
im = io.imread('./data/image/frame/denis_ADH1_27.1_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_DA27_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[57]:


im = io.imread('./data/image/mask/denis_ADH1_27.1_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_DA27_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])

# In[ ]:
im = io.imread('./data/image/frame/denis_ADH1_28.1_im.tif')
for iimage in range(np.shape(im)[0]):
    filename = './frame/1_DA28_frame_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])


# In[57]:


im = io.imread('./data/image/mask/denis_ADH1_28.1_mask.tif')
for iimage in range(np.shape(im)[0]):
    filename = './mask/1_DA28_mask_'+str(iimage)+'.tiff';
    io.imsave(filename, im[iimage])




