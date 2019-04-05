#03/04/2019
from pyMDP_functions_nb import *
import pyMDP_functions_nb as funcs

class sliding_function:
    def __init__(self,array,function,window_size=50,step=5,output_type='complex'):
        self.array=array
        self.function=function
        self.window_size=window_size
        self.step=step
        self.output_type=output_type
        
    def generate(self):
        a1=np.zeros([self.array.shape[0]+2*self.window_size,self.array.shape[1]+2*self.window_size])
        a1[self.window_size:-self.window_size,self.window_size:-self.window_size]=self.array
        x_coords=np.arange(0,self.array.shape[0],self.step)
        y_coords=np.arange(0,self.array.shape[1],self.step)
        x_dim=len(x_coords)
        y_dim=len(y_coords)
        for i in range(x_dim):
            for j in range(y_dim):
                a2=a1[x_coords[i]:2*self.window_size+x_coords[i],y_coords[j]:2*self.window_size+y_coords[j]]
                if i==0 and j==0:
                    output=np.ndarray.flatten(self.function(a2))
                    self.output_shape=self.function(a2).shape
                    transformed=np.zeros([x_dim,y_dim,len(output)],dtype=self.output_type)
                a3=self.function(a2)
                transformed[i,j]=np.ndarray.flatten(a3)
        return transformed

class full_FFT_PCA:
    def __init__(self,image,n_components=50,window_size=20,step=1):
        self.n_components=n_components
        self.window_size=window_size
        self.step=step
        self.image=image
        
    def fit(self):
        sf=sliding_function(self.image,funcs.local_fft,window_size=self.window_size,step=self.step)
        sliding_result=sf.generate()
        transformation_shape=sf.output_shape
        self.pca = decomposition.PCA(n_components=self.n_components)
#         sliding_result=sliding_result-np.min(sliding_result)
        self.fft_shape=sliding_result.shape
        Y=self.pca.fit_transform(np.real(sliding_result.reshape(sliding_result.shape[0]*sliding_result.shape[1],sliding_result.shape[2])))
        aux1=Y.reshape(sliding_result.shape[0],sliding_result.shape[1],self.n_components)
        aux2=np.swapaxes(aux1,0,2)
        self.pca_maps=np.swapaxes(aux2,1,2)
        self.pca_components_=self.pca.components_.reshape(self.n_components,
                                                     transformation_shape[0],transformation_shape[1])
class full_Radon_PCA:
    def __init__(self,image,n_components=50,window_size=20,step=1):
        self.n_components=n_components
        self.window_size=window_size
        self.step=step
        self.image=image
        
    def fit(self):
        sf=sliding_function(self.image,funcs.local_radon,window_size=self.window_size,step=self.step)
        sliding_result=sf.generate()
        transformation_shape=sf.output_shape
        self.pca = decomposition.PCA(n_components=self.n_components)
#         sliding_result=sliding_result-np.min(sliding_result)
        self.fft_shape=sliding_result.shape
        Y=self.pca.fit_transform(np.real(sliding_result.reshape(sliding_result.shape[0]*sliding_result.shape[1],sliding_result.shape[2])))
        aux1=Y.reshape(sliding_result.shape[0],sliding_result.shape[1],self.n_components)
        aux2=np.swapaxes(aux1,0,2)
        self.pca_maps=np.swapaxes(aux2,1,2)
        self.pca_components_=self.pca.components_.reshape(self.n_components,
                                                     transformation_shape[0],transformation_shape[1])        
class light_FFT_PCA:
    def __init__(self,image,n_components=50,window_size=20,step=5):
        self.n_components=n_components
        self.window_size=window_size
        self.step=step
        self.image=image
        
    def fit(self):
        sf=sliding_function(self.image,funcs.local_fft,window_size=self.window_size,step=self.step)
        sliding_result=sf.generate()
        transformation_shape=sf.output_shape
        self.pca = decomposition.PCA(n_components=self.n_components)
#         sliding_result=sliding_result-np.min(sliding_result)
        self.fft_shape=sliding_result.shape
        self.pca.fit(np.real(sliding_result.reshape(sliding_result.shape[0]*sliding_result.shape[1],sliding_result.shape[2])))
        def local_descriptor_and_pca_transform(array):
            array1=funcs.local_fft(array)
            return self.pca.transform(np.real(array1).flatten().reshape(1,-1))[0]
        fitting=sliding_function(self.image,local_descriptor_and_pca_transform,window_size=self.window_size,step=1)
        self.pca_components_=self.pca.components_.reshape(self.n_components,
                                                     transformation_shape[0],transformation_shape[1])    
        aux2=np.swapaxes(np.real(fitting.generate()),0,2)
        self.pca_maps=np.swapaxes(aux2,1,2)
        
class light_Radon_PCA:
    def __init__(self,image,n_components=50,window_size=20,step=5):
        self.n_components=n_components
        self.window_size=window_size
        self.step=step
        self.image=image
        
    def fit(self):
        sf=sliding_function(self.image,funcs.local_radon,window_size=self.window_size,step=self.step)
        sliding_result=sf.generate()
        transformation_shape=sf.output_shape
        self.pca = decomposition.PCA(n_components=self.n_components)
#         sliding_result=sliding_result-np.min(sliding_result)
        self.fft_shape=sliding_result.shape
        self.pca.fit(np.real(sliding_result.reshape(sliding_result.shape[0]*sliding_result.shape[1],sliding_result.shape[2])))
        def local_descriptor_and_pca_transform(array):
            array1=funcs.local_radon(array)
            return self.pca.transform(np.real(array1).flatten().reshape(1, -1))[0]
        fitting=sliding_function(self.image,local_descriptor_and_pca_transform,window_size=self.window_size,step=1)
        self.pca_components_=self.pca.components_.reshape(self.n_components,
                                                     transformation_shape[0],transformation_shape[1])    
        aux2=np.swapaxes(fitting.generate(),0,2)
        self.pca_maps=np.swapaxes(aux2,1,2)
        
class cross_corr_FFT_PCA:
    def __init__(self,image,n_components=50,window_size=20,step=5,step_cc=3,display=False):
        self.n_components=n_components
        self.window_size=window_size
        self.step=step
        self.step_cc=step_cc
        self.image=image
        preprocess=full_FFT_PCA(image,n_components=n_components,window_size=window_size,step=step)
        preprocess.fit()
        if display:
            for i in range(n_components):
                f,ax=plt.subplots(1,2,figsize=(8,4))
                plt.title(str(i))
                ax[0].imshow(preprocess.pca_maps[i],'bwr')
                ax[1].imshow(preprocess.pca_components_[i],'bwr')
                ax[0].set_xticks([])
                ax[1].set_xticks([])
                ax[0].set_yticks([])
                ax[1].set_yticks([])
                plt.show()
        self.pca_components_=preprocess.pca_components_
        
    def fit(self,components=[1,2,3]):
        
        def local_descriptor(array):
            a=np.abs(funcs.local_fft(array,ham_filter=True))
            b=np.abs(array2)
            return np.array([funcs.corr2(a,b)])
        for n,i in enumerate(components):
            array2=array2=self.pca_components_[i]
            sf=sliding_function(self.image,function=local_descriptor,
                             window_size=self.window_size,step=self.step_cc)
            if n==0:
                transformation_shape=sf.output_shape
                self.pca_maps=np.zeros([len(components),transformation_shape[0],transformation_shape[1]])
            self.pca_maps[n]=np.real(sf.generate())[:,:,0]
            
class wavelet_NMF:
    def __init__(self,image,n_components=3,rotation_number=10,wavelet_level=1,display=False):
        self.n_components=n_components
        self.rotation_number=rotation_number
        self.image=image
        self.n=rotation_number
        xx=180
        ddata=[]
        i=0
        for rotation in range(0,xx,self.n):

            kart=self.image
            kart=np.pad(kart,pad_width=256,mode='wrap')
            cut=int(self.image.shape[0]*2**0.5/2)
            rotation_matrix = cv2.getRotationMatrix2D((kart.shape[0]/2, (kart.shape[1]/2)), rotation, 1)
            img_rotation = cv2.warpAffine(kart, rotation_matrix, (kart.shape[0], kart.shape[1]))
            img_rotation=img_rotation-np.min(img_rotation)
            img_rotation=img_rotation/np.max(img_rotation)
            S=pywt.swt2(img_rotation, 'bior5.5', 4, start_level=0, axes=(-2, -1))
            img1=S[wavelet_level][1][0]
            rotation_matrix = cv2.getRotationMatrix2D((img1.shape[0]/2, (img1.shape[1]/2)), -rotation, 1)
            img_rotation = cv2.warpAffine(img1, rotation_matrix, (img1.shape[0], img1.shape[1]))
            img_rotation=img_rotation[int(img_rotation.shape[0]/2-self.image.shape[0]/2):int(img_rotation.shape[0]/2+self.image.shape[0]/2),int(img_rotation.shape[1]/2-self.image.shape[1]/2):int(img_rotation.shape[1]/2+self.image.shape[1]/2)]
           
            wdata=np.abs(img_rotation)
            if display:
                fig,ax=plt.subplots(1,2,figsize=(15,30))
                plt.title(str(rotation)+' '+str(i))
                ax[0].imshow(wdata,cmap='bwr',vmin=-0.5,vmax=0.5)
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[1].imshow(self.image,cmap='bwr')
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                plt.plot([200,200+100*np.cos(rotation*np.pi / 180)],[200,200+100*np.sin(rotation*np.pi / 180)],color='k')
                ax[1].scatter([200],[200],color='r')
                ax[1].scatter([200+100*np.cos(rotation*np.pi / 180)],[200+100*np.sin(rotation* np.pi / 180)],color='k')
                
                plt.show()
            i=i+1
            ddata+=[img_rotation]
        self.i=i
        ddata=np.array(ddata)
        ddata=np.swapaxes(ddata,0,2)
        ddata=np.swapaxes(ddata,0,1)
        self.data=ddata
        
    def fit(self):
        nmf = decomposition.NMF(n_components=self.n_components)
        preprocess=nmf.fit_transform(np.abs(self.data.reshape(self.data.shape[0]*self.data.shape[1],self.data.shape[2])))
        cube=[]
        for i in range(3):
            dddata=np.abs(preprocess.reshape( self.image.shape[0], self.image.shape[1],self.n_components)[:,:,i])
            cube+=[dddata]
        self.pca_maps=np.array(cube)
        


