#03/04/2019
from pyMDP_map_generators import *

class multidomain_picture:
    def __init__(self,filename):
        self.data='None'
        self.filename=filename
        self.labels="None"
        self.original_image='None'
        self.scale=500
        self.pixel_size=512
        self.maps=dict()
       
    def show_labels(self):
        labels=loadibw(self.filename)['wave']['labels'][2][1:]
        for i in range(len(labels)):
            print(labels[i].decode('utf-8'))
            
    def load_data(self,scale=500,display=True):
        self.scale=scale
        self.data=loadibw(self.filename)['wave']['wData'] 
        self.channels=loadibw(self.filename)['wave']['labels']
        if display:
            f,ax=plt.subplots(1,self.data.shape[2],figsize=(self.data.shape[2]*5,6))
            for i in range(self.data.shape[2]):
                a=funcs.flatten_1d(self.data[:,:,i].T)[::-1]
                self.pixel_size=a.shape[0]
                b=self.channels[2][1:][i]
                ax[i].set_title(str(b.decode('utf-8'))[:-7]+', index = '+str(i))
                ax[i].imshow(a)
            for l in range(self.data.shape[2]):
                ax[l].set_xticks([])
                ax[l].set_yticks([])
            plt.tight_layout()
            plt.show()
    
    def select_original_image(self,index,flatten=True,deg=1,remove_horizontal_jitter=False,level=1,display=True,j=300,k=100):
        if flatten:
            array=funcs.flatten_1d(self.data[:,:,index].T,deg=deg)[::-1]
        else:
            array=self.data[:,:,index].T[::-1]
        if remove_horizontal_jitter:
            level=level
            import pywt
            A=pywt.wavedec2(array, 'db2',level=level)
            A = list(A)
            for i in range(1,level+1):
                A[i] = list(A[i])
                A[i][0]=A[i][0]*0
            array=pywt.waverec2(A, 'db2')
        if display:
            f,ax=plt.subplots(1,5,figsize=(30,6))
            ax[0].imshow(array,'bwr')
            j=300
            k=100
            w=30
            c=array[j-w:j+w,k-w:k+w]
            ax[1].imshow(c,'bwr')
            ax[2].imshow(np.real(funcs.local_fft(c,ham_filter=True)),cmap='binary',interpolation='gaussian') 
            ax[3].imshow(np.imag(funcs.local_fft(c,ham_filter=True)),cmap='binary',interpolation='gaussian')
            ax[4].imshow(np.abs(funcs.local_fft(c,ham_filter=True)),cmap='binary',interpolation='gaussian')
            for l in range(5):
                ax[l].set_xticks([])
                ax[l].set_yticks([])
            ax[0].set_title(str(self.channels[2][1:][index].decode('utf-8')[:-7])+', index = '+str(index))
            ax[1].set_title('Cutaway')
            ax[2].set_title('Real part of FFT')
            ax[3].set_title('Imaginary part of FFT')
            ax[4].set_title('Absolute FFT')
            plt.show()
        self.original_image=array
           
    def generate_maps(self,method,window_size=20,step=5,n_components=10,output_type='complex',display=True,components=[1,2,3],
                 rotation_number=10,wavelet_level=1):
        image=self.original_image
        params=[n_components,window_size,step,output_type,display,rotation_number,wavelet_level]
        classifiers=[full_FFT_PCA,full_Radon_PCA,light_FFT_PCA,light_Radon_PCA,cross_corr_FFT_PCA,wavelet_NMF]
        relevant_parameters=[np.array([0,1,2]),
                             np.array([0,1,2]),
                             np.array([0,1,2]),
                             np.array([0,1,2]),
                             np.array([0,1,2,4]),
                             np.array([0,5,6,4])]
        if method not in classifiers:
            print("No such method!")
        else:
            clf=classifiers[classifiers.index(method)](image,*[params[i] for i in relevant_parameters[classifiers.index(method)]])
            if method!=cross_corr_FFT_PCA:
                clf.fit()
            else:
                clf.fit(components=components)
        self.maps[method]=clf.pca_maps
        if method==full_FFT_PCA or method==full_Radon_PCA:
            if display:
                for i in range(n_components):
                    fig,ax=plt.subplots(1,2,figsize=(8,4))
                    plt.title(str(i))
                    ax[0].imshow(clf.pca_maps[i],'bwr')
                    ax[1].imshow(clf.pca_components_[i],'bwr')
                    ax[0].set_xticks([])
                    ax[1].set_xticks([])
                    ax[0].set_yticks([])
                    ax[1].set_yticks([])
                    plt.show()
    
    def display_maps(self,method):
        if method not in self.maps.keys():
            print('Please calculate the maps first')
        else:
            array=self.maps[method]
        for i in range(len(array)):
            fig,ax=plt.subplots(1,2,figsize=(16,8))
            plt.title('Component '+str(i))
            ax[0].imshow(self.original_image,'binary')
            ax[1].imshow(array[i],'inferno',alpha=1)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            plt.show()
            
    def pick_background(self,background_type,method=full_FFT_PCA,index=0):
        background_types = ['PCA 0', 'Other channel','Zeroes']
        if background_type not in background_types:
            raise ValueError("Invalid background type. Expected one of: %s" % background_types)    
        if background_type=='Zeroes':
            self.background=np.zeros_like(self.original_image)
        elif background_type=='Other channel':
            self.background=funcs.flatten_1d(self.data[:,:,index].T,deg=1)[::-1]
        elif background_type=='PCA 0':
            self.background=self.maps[method][0]
            
            
    def select_maps(self,method=full_FFT_PCA, indices=[1,2,3],use_background=True,
                normalize=True, blur=True, invert_backgound=False,blur_indices=[5,5,5,5],otsu=True,display=True,radius=200):
        self.abundances=np.array([self.maps[method][i] for i in indices])
        if use_background==True:
            self.use_background=True
            back=cv2.resize(self.background,(self.maps[method][0].shape[0],self.maps[method][0].shape[1]))
            if invert_backgound==True:
                back=back.max()-back
            if otsu:
                back=back.max()-back
                array=(funcs.norm(back)*255).astype('uint8')
                radius=radius
                selem = disk(radius)
                local_otsu = rank.otsu(array, selem)
                diff_ann=cv2.GaussianBlur(array-local_otsu,(1,1),1)
                back=diff_ann
            self.abundances=np.concatenate([back.reshape(1,back.shape[0],back.shape[1]),self.abundances])

        else:
            self.use_background=False
        for i in range(len(self.abundances)):
            self.abundances[i]=np.abs(self.abundances[i])
        if blur==True:
            for i in range(len(self.abundances)):
                kernel = np.ones((blur_indices[i],blur_indices[i]),np.float32)/blur_indices[i]
                self.abundances[i]=cv2.filter2D(self.abundances[i],-1,kernel)
        if normalize==True:
            for i in range(len(self.abundances)):
                self.abundances[i]=funcs.norm(self.abundances[i])
        if display:
            fig,ax=plt.subplots(1,self.abundances.shape[0],figsize=(self.abundances.shape[0]*4,4))
            for i in range(self.abundances.shape[0]):
                alpha=1
                ax[i].imshow(self.abundances[i],'Reds',alpha=alpha)
                ax[i].set_xlabel(str(i))
                ax[i].set_xticks([])
                ax[i].set_yticks([])
            plt.show()



    def classify_by_NFINDR(self,n=3,threshold=0.4):
        nfindr = eea.NFINDR()
        if self.use_background==True:
            U = nfindr.extract(np.swapaxes(np.swapaxes(self.abundances,0,2),0,1), n+1, normalize=True)
        else:
            U = nfindr.extract(np.swapaxes(np.swapaxes(self.abundances,0,2),0,1), n, normalize=True)
        self.U=np.sort(U,axis=0)
        
        self.labels=np.zeros_like(self.abundances[0])
        for i in range(self.abundances.shape[1]):
            for j in range(self.abundances.shape[2]):
                if self.use_background==True:
                    self.labels[i,j]=funcs.find_nearest(np.array([U[i] for i in range(len(U))]),self.abundances[:,i,j])
                else:
                    self.labels[i,j]=funcs.find_nearest(np.array([U[i] for i in range(len(U))]),self.abundances[:,i,j])+1
        self.ds_labels=self.labels
        self.labels=cv2.resize(np.array(self.labels,dtype='uint8'),dsize=(self.original_image.shape[0],self.original_image.shape[1]),interpolation = cv2.INTER_NEAREST)
                
    def classify_by_coordinate(self,n=3,threshold=0.4):

        self.nc=np.eye(n)
        self.labels=np.zeros_like(self.abundances[0])
        for i in range(self.abundances.shape[1]):
            for j in range(self.abundances.shape[2]):
                if self.use_background==True:
                    self.labels[i,j]=funcs.find_nearest(np.array([self.nc[k] for k in range(len(self.nc))]),self.abundances[:,i,j][1:])+1
                    if self.abundances[:,i,j][0]>threshold:
                        self.labels[i,j]=0
                else:
                    self.labels[i,j]=funcs.find_nearest(np.array([self.nc[k] for k in range(len(self.nc))]),self.abundances[:,i,j])+1
        self.ds_labels=self.labels
        self.labels=cv2.resize(np.array(self.labels,dtype='uint8'),dsize=(self.original_image.shape[0],self.original_image.shape[1]),interpolation = cv2.INTER_NEAREST)

    def  classify_by_cluster_centroids(self,n_clusters=16,n=3,display=True,sparcity=10,threshold=0.4,method='Coordinate'):
        self.km = KMeans(n_clusters=n_clusters, random_state=0).fit(self.abundances.reshape(self.abundances.shape[0],self.abundances.shape[1]*self.abundances.shape[2]).T)
        methods = ['Coordinate', 'NFINDR']
        if method not in methods:
            raise ValueError("Invalid method type. Expected one of: %s" % methods)
        nfindr = eea.NFINDR()
        U = nfindr.extract(np.swapaxes(np.swapaxes(self.abundances,0,2),0,1), 3, normalize=True)
        self.U=np.sort(U,axis=0)
        colors=np.ones_like(self.km.labels_)
        count=[]
        if method=="Coordinate":
            if self.use_background==True:
                hole_comp=np.zeros([n+1])
                hole_comp[0]=1
                colors[self.km.labels_==funcs.find_nearest(self.km.cluster_centers_ ,hole_comp)]=0
                for i in range(max(self.km.labels_)+1):
                    if self.km.cluster_centers_[i,0]>threshold:
                        colors[self.km.labels_==i]=0
                    else:
                        true_color=np.argmax(self.km.cluster_centers_[i][1:])+1
                        colors[self.km.labels_==i]=true_color
                        count+=[i]
            else:
                for i in range(max(self.km.labels_)+1):
                    true_color=np.argmax(self.km.cluster_centers_[i])+1
                    colors[self.km.labels_==i]=true_color
                    count+=[i]
        elif method=="NFINDR":
            print("Please use coordinate method for now")
#             if self.use_background==True:
#                 idx=np.argmax(U[:,0])
#                 hole_comp=U[idx]
#                 colors[self.km.labels_==funcs.find_nearest(self.km.cluster_centers_ ,hole_comp)]=0
#                 for i in range(max(self.km.labels_)+1):
#                     true_color=funcs.find_nearest(np.array([U[i] for i in range(len(U)) if i!=idx]),self.km.cluster_centers_[i][1:] )+1
#                     colors[self.km.labels_==i]=true_color

#                 for i in range(max(self.km.labels_)+1):
#                     true_color_b=np.argmin([(np.sum(self.km.cluster_centers_[i][0]-hole_comp[0])**2)**0.5,(np.sum(self.km.cluster_centers_[i][0]-[0,0,0,0][0])**2)**0.5])
#                     if true_color_b==0:
#                         colors[self.km.labels_==i]=true_color_b
#                     count+=[i]    
        self.ds_labels=np.array(colors.reshape(self.abundances.shape[1],self.abundances.shape[2]),dtype='uint8')
        self.labels=cv2.resize(np.array(colors.reshape(self.abundances.shape[1],self.abundances.shape[2]),dtype='uint8'),dsize=(self.original_image.shape[0],self.original_image.shape[1]),interpolation = cv2.INTER_NEAREST)
        
    def inspect_clustering(self,indices=[1,2,3],sparcity=10,alpha=0.2,s=1,azim=45,elev=45,show_initial=False,show_kmeans=False,show_nfindr=False):
        
        xs=np.ndarray.flatten(self.abundances[indices[0]])[::sparcity]
        ys=np.ndarray.flatten(self.abundances[indices[1]])[::sparcity]
        zs=np.ndarray.flatten(self.abundances[indices[2]])[::sparcity]
        if show_initial==True:
         
            colors=self.km.labels_
        else:
            colors=self.ds_labels.reshape(self.ds_labels.shape[0]*self.ds_labels.shape[1])
        fig = plt.figure()
        fig.set_size_inches(7,7)
        ax = fig.add_subplot(111, projection='3d',azim=azim, elev=elev)
        ax.scatter(xs, ys, zs,alpha=alpha,s=s,c=colors[::sparcity],cmap='hsv',vmin=0,vmax=np.max(colors)+1)
        
        if show_nfindr:
            xs_nf=self.U[:,indices[0]]
            ys_nf=self.U[:,indices[1]]
            zs_nf=self.U[:,indices[2]]
            for xx,yy,zz in zip(xs_nf,ys_nf,zs_nf): ax.plot([xx,xx],[yy,yy],[0,zz], '-',c='r')
            for xx,yy,zz in zip(xs_nf,ys_nf,zs_nf): ax.plot([0,xx],[yy,yy],[zz,zz], '-',c='r')
            for xx,yy,zz in zip(xs_nf,ys_nf,zs_nf): ax.plot([xx,xx],[0,yy],[zz,zz], '-',c='r')
            ax.scatter(xs_nf, ys_nf, zs_nf,alpha=1,s=300,c='r',vmin=0,vmax=np.max(colors)+1)
        if show_kmeans:
            xs_km=np.ndarray.flatten(self.km.cluster_centers_[:,indices[0]])
            ys_km=np.ndarray.flatten(self.km.cluster_centers_[:,indices[1]])
            zs_km=np.ndarray.flatten(self.km.cluster_centers_[:,indices[2]])
            for xx,yy,zz in zip(xs_km,ys_km,zs_km): ax.plot([xx,xx],[yy,yy],[0,zz], '-',c='k')
            for xx,yy,zz in zip(xs_km,ys_km,zs_km): ax.plot([0,xx],[yy,yy],[zz,zz], '-',c='k')
            for xx,yy,zz in zip(xs_km,ys_km,zs_km): ax.plot([xx,xx],[0,yy],[zz,zz], '-',c='k')
            ax.scatter(xs_km, ys_km, zs_km,alpha=1,s=300,c='k',vmin=0,vmax=np.max(colors)+1)
        ax.tick_params(axis='both', which='major', pad=10)
        plt.tight_layout()
        plt.show()
    
    def display_results(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(7,7) 
        plt.title('Classified image')
        ax.imshow(self.original_image,'binary')
        ax.imshow(self.labels,cmap='hsv',vmin=0,vmax=np.max(self.labels)+1,alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
    def radial_profile(self,data, center,angle_selection='none'):
        
        if angle_selection=='none':
            selector=np.ones_like(data)
        else:
            array=np.vstack([np.zeros([data.shape[0]//2,data.shape[0]]),np.ones([data.shape[0]//2,data.shape[1]])])
            rotation=angle_selection[0]
            img_rotation_1 = scipy.ndimage.interpolation.rotate(array, angle=rotation,mode ='nearest',reshape =False)*2-1
            array=np.vstack([np.zeros([data.shape[0]//2,data.shape[0]]),np.ones([data.shape[0]//2,data.shape[1]])])
            rotation=180+angle_selection[1]
            img_rotation_2 = scipy.ndimage.interpolation.rotate(array, angle=rotation,mode ='nearest',reshape =False)*2-1
            selector=np.rint(img_rotation_1)*np.rint(img_rotation_2)
            selector=(selector+1)/2
        data=data*selector
        y,x = np.indices((data.shape)) # first determine radii of all pixels
        r = np.sqrt((x-center[0])**2+(y-center[1])**2)
        ind = np.argsort(r.flat) # get sorted indices
        sr = r.flat[ind] # sorted radii
        sim = data.flat[ind] # image values sorted by radii
        ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
        # determining distance between changes
        deltar = ri[1:] - ri[:-1] # assume all radii represented
        rind = np.where(deltar)[0] # location of changed radius
        nr = rind[1:] - rind[:-1] # number in radius bin
        csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
        tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
        radialprofile = tbin/nr # the answer
        return radialprofile,selector    
    
    def generate_angular_profile(self,array,fft_min=10,fft_max=500,step=1,aver=1):
        a=np.zeros([len(list(range(0,180,step)))])
        for i in range(0,180,step):
            b,s=self.radial_profile(array,center=(array.shape[0]//2,array.shape[1]//2),angle_selection=[i,i+aver])
            b[:fft_min]=0
            b[fft_max:]=0
            a[i]=np.sum(b)
        return a

    def analyze_cluster(self,direction,domain,display=True,size_cutoff=100,fft_cutoff_min=50,fft_cutoff_max=350,ham_filter=True,angle_selection='none',export_rad=False,vmax='default',vmax_angle='default',step=1,aver=2):
        labeled_domains, _ = ndi.label(self.labels==direction)
        for i in range(1,np.max(labeled_domains)+1):
            if np.sum(labeled_domains==i)<size_cutoff:
                labeled_domains[labeled_domains==i]=0
        recalc=np.unique(labeled_domains)
        for n,i in enumerate(recalc):
            labeled_domains[labeled_domains==i]=n
        b=np.copy(self.original_image)
        b[labeled_domains!=domain]=0
        c=np.copy(b)
        b[labeled_domains==domain]=1
        ang_prof=self.generate_angular_profile(np.abs(funcs.local_fft(c,ham_filter=ham_filter)),
                                     fft_min=fft_cutoff_min,fft_max=fft_cutoff_max,step=1,aver=1)
        if display:
            fig,ax=plt.subplots(1,5,figsize=(30,6))
            ax[0].imshow(np.copy(self.original_image),'binary')
            ax[0].imshow(self.labels,cmap='hsv',vmin=0,vmax=np.max(self.labels)+1,alpha=0.3)
            ax[1].imshow(labeled_domains,'nipy_spectral')
            ax[2].imshow(b,'bwr')
            if vmax=='default':
                im=ax[3].imshow(np.abs(funcs.local_fft(c,ham_filter=ham_filter)),
                         vmax=np.max(np.abs(funcs.local_fft(c,ham_filter=ham_filter))),cmap='binary',interpolation='gaussian')
            else:
                im=ax[3].imshow(np.abs(funcs.local_fft(c,ham_filter=False)),
                         vmax=vmax,cmap='binary',interpolation='gaussian') 
            divider = make_axes_locatable(ax[3])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, ax=ax[3],cax=cax)
            
            
            
            ax[4].plot(np.linspace(0,180,len(ang_prof)),ang_prof)
            for l in range(4):
                ax[l].set_xticks([])
                ax[l].set_yticks([])
            plt.show()
        domain_area=np.sum(b)/(self.pixel_size*self.pixel_size)*(self.scale*self.scale)
        b[labeled_domains!=domain]=-1
        border=(-b*np.roll(b,1,axis=1))+(-b*np.roll(b,1,axis=0))>-1
        domain_border_no_edge=np.sum(border[1:-1,1:-1])*self.scale/self.pixel_size
        domain_border=np.sum(border)*self.scale/self.pixel_size
        
        b=np.copy(self.original_image)
        b[labeled_domains!=domain]=0
        center= (int(b.shape[0]/2), int(b.shape[1]/2))
        array=np.abs(funcs.local_fft(b,ham_filter=ham_filter))
        array=array-np.min(array)
        array=array/np.max(array)
        rad,selector = self.radial_profile(array, center,angle_selection=angle_selection)
        rad[:fft_cutoff_min]=0
        rad[fft_cutoff_max:]=0
        if display:
            print(np.argmax(rad))
            fig,ax=plt.subplots(1,2,figsize=(12,6))
            ax[0].plot(rad,color='b')
            ax[0].axvline(np.argmax(rad))
            ax[0].set_xlabel('Spacing')
            ax[0].set_ylabel('Intensity')
            if vmax_angle=='default':
                im=ax[1].imshow(array*selector,
                         cmap='binary',interpolation='gaussian')
            else:
                im=ax[1].imshow(array*selector,vmax=vmax_angle,
                         cmap='binary',interpolation='gaussian')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, ax=ax[1],cax=cax)
            plt.tight_layout()
            plt.show()
          
        domain_spacing=(self.pixel_size)/np.argmax(rad)*self.scale/self.pixel_size
        
            
        if export_rad==False:  
            return np.max(labeled_domains),domain,domain_area,domain_border_no_edge,domain_border,domain_spacing
        else:
            return np.max(labeled_domains),domain,domain_area,domain_border_no_edge,domain_border,domain_spacing,self.radial_profile(array, center,angle_selection=angle_selection)[0],ang_prof
    
    def gather_staticstics(self,display=False,size_cutoff=100,fft_cutoff_min=50,fft_cutoff_max=350,
                           ham_filter=False,angle_selection='none',export_rad=False,n_maps=3,
                           vmax='default',vmax_angle='default'):

        domain_dict={}
        k=0
        for direction in range(1,n_maps+1):
            direction_list=[]
            labeled_domains, _ = ndi.label(self.labels==direction)
            for i in range(1,np.max(labeled_domains)+1):
                if np.sum(labeled_domains==i)<100:
                    labeled_domains[labeled_domains==i]=0
            recalc=np.unique(labeled_domains)
            for n,i in enumerate(recalc):
                labeled_domains[labeled_domains==i]=n
            if angle_selection=='none':
                for domain in range(1,np.max(labeled_domains)+1):
                    direction_list+=[self.analyze_cluster(direction=direction,
                                                          domain=domain,display=display,
                                                          size_cutoff=size_cutoff,
                                                          fft_cutoff_min=fft_cutoff_min,fft_cutoff_max=fft_cutoff_max,
                                                          ham_filter=ham_filter,export_rad=export_rad,
                                                          vmax=vmax,vmax_angle=vmax_angle)]
            else:
                for domain in range(1,np.max(labeled_domains)+1):
                    direction_list+=[self.analyze_cluster(direction=direction,
                                                          domain=domain,display=display,
                                                          size_cutoff=size_cutoff,
                                                          fft_cutoff_min=fft_cutoff_min,fft_cutoff_max=fft_cutoff_max,
                                                          ham_filter=ham_filter,
                                                          vmax=vmax,vmax_angle=vmax_angle,
                                                          angle_selection=angle_selection[k],export_rad=export_rad)]
                    
            domain_dict[direction]=direction_list
            k=k+1
        self.domain_dict=domain_dict
        return domain_dict
    
class HIM_BB_picture:
    def __init__(self,folder,name,folder_to_save='default',saveit=False):
        self.path=folder+name
        self.name=name[:-4]
        self.saveit=saveit
        if folder_to_save=='default':
            self.folder_to_save=folder
        else:
            self.folder_to_save=folder_to_save
           
    def load_data(self,display=True,limits=False,saveit=False):
        a0=plt.imread(self.path)
        a0=cv2.cvtColor(a0,cv2.COLOR_RGB2GRAY)
        a0=a0[:a0.shape[1]]
        if limits!=False:
            a0=a0[limits[0][0]:limits[0][1],limits[1][0]:limits[1][1]]
        self.data=a0
        
        if display:
            fig,ax=plt.subplots(1,1,figsize=(5,5))
            ax.imshow(a0,cmap='gray')
            if saveit:
                plt.savefig(self.folder_to_save+self.name+'_original'+'.png', type='png',bbox_inches='tight')
            plt.show()
    def generate_markers(self,radius=10,thres=80,display=True,saveit=False):
        image=self.data
        radius=radius
        selem = disk(radius)
        local_otsu = rank.otsu(self.data, selem)
        diff_ann=cv2.GaussianBlur(image-local_otsu,(5,5),5)

        thres=thres
        markers=np.zeros_like(diff_ann)  
        markers[diff_ann>thres]=1
        skeleton = skeletonize(markers)
        self.markers=markers
        self.skeleton=skeleton
        if display:
            fig,ax=plt.subplots(1,5,figsize=(25,5))
            ax[0].imshow(self.data,cmap='gray')
            im=ax[1].imshow(diff_ann,cmap='gray')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, ax=ax[1],cax=cax)
            h=ax[2].hist(diff_ann.flatten(),bins=100)
            ax[3].imshow(markers,'binary')
            ax[4].imshow(skeleton,'gray')
            plt.tight_layout()
            if saveit:
                plt.savefig(self.folder_to_save+self.name+'_markers'+'.png', type='png',bbox_inches='tight')
            plt.show()
            
    def calculate_skeleton_orientations(self,frame_size=7,display=True,saveit=False):
        expanded_frame=np.zeros([self.data.shape[0]+100,self.data.shape[1]+100])
        expanded_frame[50:self.data.shape[0]+50,50:self.data.shape[1]+50]=self.skeleton
        matrixc=np.zeros_like(expanded_frame)
        labeled1=np.where(expanded_frame!=0)
        vmin=-np.pi/2
        vmax=np.pi/2
        frame_size=frame_size
        for p in tqdm(range(len(labeled1[0]))):
            point=p
            X=labeled1[0][point]
            Y=labeled1[1][point]
            local_frame=expanded_frame[X-frame_size:X+frame_size,Y-frame_size:Y+frame_size]
            local_frame=local_frame.astype(np.int)
            props = regionprops(local_frame)
            rotation=props[0].orientation
            matrixc[X,Y]=rotation
        self.orientations=matrixc
        if display:
            fig,ax=plt.subplots(1,1,figsize=(5,5))
            ax.imshow(matrixc,cmap='hsv')
            ax.imshow(np.dstack([np.ones_like(matrixc),np.ones_like(matrixc),np.ones_like(matrixc),(matrixc==0).astype(int)]),cmap='gray')
            plt.xlim(50,self.data.shape[0]+50)
            plt.ylim(self.data.shape[1]+50,50)
            plt.savefig(self.folder_to_save+self.name+'_orientations'+'.png', type='png',bbox_inches='tight')
            plt.show()
            
            f,ax=plt.subplots(figsize=(5,5))
            self.orientations_hist=ax.hist(np.ndarray.flatten(matrixc[np.where(matrixc!=0)]),bins=90,color='b')
            ax.set_xlabel('Angle, rad')
            ax.set_ylabel('Count')
            plt.tight_layout()
            if saveit:
                plt.savefig(self.folder_to_save+self.name+'_hist'+'.png', type='png',bbox_inches='tight')
            plt.show()
    
    def create_domains(self,window_size=30,step=3,n_components=3,component=0,display=True,saveit=False):
        f=multidomain_picture(filename='1') 
        f.original_image=self.markers
        f.generate_maps(cross_corr_FFT_PCA,window_size=30,step=3,n_components=3,
                output_type='complex',
                components=[component],
                rotation_number=10,wavelet_level=1,
                display=True)
        self.domain_probabilities=np.abs(f.maps[cross_corr_FFT_PCA][0])    
        if display:
            fig,ax=plt.subplots(1,3,figsize=(15,5))
            ax[0].imshow(self.markers,cmap='binary')
            ax[0].imshow(self.domain_probabilities,cmap='inferno',alpha=0.3)
            ax[1].imshow(self.domain_probabilities,cmap='inferno')
            ax[2].imshow(self.domain_probabilities>0.4,cmap='inferno')
            if saveit:
                plt.savefig(self.folder_to_save+self.name+'_domain_probabilities'+'.png', type='png',bbox_inches='tight')
            plt.show()
            
    def assign_domains(self,radius=50, thres=40,display=True,saveit=False): 
        image=(-self.domain_probabilities*255).astype('uint8')
        radius=radius
        selem = disk(radius)
        local_otsu = rank.otsu(image, selem)
        diff_ann=image-local_otsu
        thres=thres
        markers=np.zeros_like(diff_ann)  
        markers[diff_ann<thres]=1
        
        size_cutoff=200
        labeled_domains, _ = ndi.label(markers==0)
        for i in range(1,np.max(labeled_domains)+1):
            if np.sum(labeled_domains==i)<size_cutoff:
                labeled_domains[labeled_domains==i]=0
        recalc=np.unique(labeled_domains)
        self.unlabelled_domains=markers
        self.labeled_domains=labeled_domains
        for n,i in enumerate(recalc):
            labeled_domains[labeled_domains==i]=n
        if display:
            fig,ax=plt.subplots(1,5,figsize=(25,5))
            ax[0].imshow(self.domain_probabilities,cmap='gray')
            im=ax[1].imshow(diff_ann,cmap='gray')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, ax=ax[1],cax=cax)
            h=ax[2].hist(diff_ann.flatten(),bins=100)
            ax[3].imshow(markers,'binary')
            ax[4].imshow(labeled_domains,'nipy_spectral')
            plt.tight_layout()
        if saveit:
            plt.savefig(self.folder_to_save+self.name+'_assigned_domains'+'.png', type='png',bbox_inches='tight')    
        plt.show()
                    
    def adjust_domains(self,blur_erosion=5,blur_dilation=15,saveit=False):
        kernel = np.ones((blur_erosion,blur_erosion),np.uint8)
        erosion = cv2.dilate(self.unlabelled_domains,kernel,iterations = 1)
        size_cutoff=200
        labeled_domains, _ = ndi.label(erosion==0)
        for i in range(1,np.max(labeled_domains)+1):
            if np.sum(labeled_domains==i)<size_cutoff:
                labeled_domains[labeled_domains==i]=0
        recalc=np.unique(labeled_domains)
        for n,i in enumerate(recalc):
            labeled_domains[labeled_domains==i]=n
        if display:
            fig,ax=plt.subplots(1,2,figsize=(10,5))
            ax[0].imshow(labeled_domains,cmap='nipy_spectral')
        
        kernel = np.ones((blur_dilation,blur_dilation),np.uint8)
        labeled_domains2=np.zeros_like(labeled_domains)
        for i in range(1, labeled_domains.max()+1):
            aux=np.zeros_like( labeled_domains)
            aux[ labeled_domains==i]=1
            dilation =cv2.dilate(aux.astype('uint8'),kernel,iterations = 1)
            labeled_domains2[dilation ==1]=i
        self.adjusted_domains=labeled_domains2
        if display:
            ax[1].imshow(labeled_domains2,cmap='nipy_spectral')
            if saveit:
                plt.savefig(self.folder_to_save+self.name+'_adjusted_domains'+'.png', type='png',bbox_inches='tight') 
            plt.show()
            
    def show_domain_overlay(self,saveit=False):
        fig,ax=plt.subplots(1,1,figsize=(10,10))
        ax.imshow(self.data,cmap='gray')
        ax.imshow(self.adjusted_domains,'nipy_spectral',alpha=0.3)
        if saveit:
            plt.savefig(self.folder_to_save+self.name+'_domain_overlay'+'.png', type='png',bbox_inches='tight') 
        plt.show()
        
    def FFT_analysis(self,indices=[[0,256],[0,256]],zoom=True,FFT_FOV=50,vmax=5e4,aver=20,fft_limits=[[1,7],[7,20]],saveit=False):
        a0=self.data[indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]
        A=np.abs(funcs.local_fft(a0,ham_filter=True))
        A2=np.abs(A)[A.shape[0]//2-FFT_FOV:A.shape[0]//2+FFT_FOV,A.shape[1]//2-FFT_FOV:A.shape[1]//2+FFT_FOV]
        if display:
            fig,ax=plt.subplots(1,len(fft_limits)+2,figsize=(5*(len(fft_limits)+2),5))
            im=ax[0].imshow(A2,vmax=vmax,cmap='binary',interpolation='Gaussian')
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, ax=ax[0],cax=cax)
        radial_profile=funcs.radial_profile(A2,center=(A2.shape[0]//2,A2.shape[1]//2))[0][1:]
        angular_profiles=[]
        for i in range(len(fft_limits)):
            angular_profiles+=[funcs.generate_angular_profile(A2,fft_min=fft_limits[i][0],fft_max=fft_limits[i][1],aver=aver)]
        self.fft_results={}
        self.fft_results['radial_profile']=radial_profile
        self.fft_results['angular_profiles']=angular_profiles
        self.fft_results['FFT']=A2
        if display:
            ax[1].plot(radial_profile)
            for i in range(len(fft_limits)):
                ax[2+i].plot(angular_profiles[i])
            plt.tight_layout()
            if saveit:
                plt.savefig(self.folder_to_save+self.name+'_fft'+'.png', type='png',bbox_inches='tight') 
            plt.show()
            
class spectral_correlation:
    def __init__(self,data):
        self.data=data
        self.len=data.shape[0]
        
    def display_data(self,display=True):
        if display:
            fig,ax=plt.subplots(1,1,figsize=(10,10))
            s=ax.imshow(self.data,'bwr',vmin=-1,vmax=1)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()        

            fig,ax=plt.subplots(1,1,figsize=(5,5))
            s=plt.hist(self.data.flatten(),bins=100)
            plt.tight_layout()
            plt.show()
        
    def calculate_similarity(self,power=-0.1,display=False,vmin=0,vmax=1):    
        self.similarity = np.exp( power* self.data/self.data.std())
#         self.similarity=(1-self.data)**0.5
        if display:
            fig,ax=plt.subplots(1,1,figsize=(10,10))
            s=ax.imshow(self.similarity,'bwr',vmin=vmin,vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()
            
            fig,ax=plt.subplots(1,1,figsize=(5,5))
            s=plt.hist(self.similarity.flatten(),bins=100)
            plt.tight_layout()
            plt.show()
            
            
            
    def calculate_labels(self,power=-0.1,display=False,algorithm='default',**kwargs):   
        if algorithm=='default':
            model = AffinityPropagation(affinity='precomputed')
            model.fit(self.similarity)
            N=np.max(model.labels_)+1
            model = SpectralClustering(N)
            model.fit(self.similarity)
        else:
            model = algorithm(**kwargs)
            model.fit(self.similarity)
        self.labels=model.labels_
        if display:         
            bar=np.zeros([10,self.len])
            for i in range(10):
                bar[i]=model.labels_
            fig,ax=plt.subplots(1,1,figsize=(10,10))
            dd=ax.imshow(bar,cmap='rainbow',vmin=0,vmax=np.max(bar))
            ax.set_aspect(self.len/100)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.show()
            
    def visualize(self,s=50,azim=25,elev=45,alpha=0.3,display=False,cmap='rainbow',zoom=False):
        c=self.labels
        M=np.zeros([self.len,self.len])
        for i in range(self.len):
            for j in range(self.len):
                M[i,j]=self.similarity[0,j]**2+self.similarity[i,0]**2-self.similarity[i,j]**2
        Ss,U=np.linalg.eig(M)
        S=np.eye(self.len,dtype='complex')
        for i in range(self.len):
            S[i,i]=Ss[i]
        X=np.matmul(U,np.real(S**0.5))
        pca=PCA()
        data=pca.fit_transform(X)
        self.kpi=np.sum((scipy.spatial.distance_matrix(x=data,y=data)-self.similarity)**2)**0.5/self.len/self.len/np.mean(self.similarity)
        recon=np.zeros([self.len,self.len])
        for i in range(self.len):
            for j in range(self.len):
                recon[i,j]=np.sum((data[i,:3]-data[j,:3])**2)**0.5
        self.transformed_data=data
        if display:
            fig = plt.figure()
            fig.set_size_inches(10,15)
            ax = fig.add_subplot(321, projection='3d',azim=azim, elev=elev)
            ax.scatter(data[:,0], data[:,1], data[:,2],alpha=alpha,s=s,c=c,cmap='rainbow',vmin=0,vmax=np.max(c))
            if zoom!=False:
                ax.set_xlim(np.mean(data[:,0])-zoom*np.std(data[:,0]),np.mean(data[:,0])+zoom*np.std(data[:,0]))
                ax.set_ylim(np.mean(data[:,1])-zoom*np.std(data[:,1]),np.mean(data[:,1])+zoom*np.std(data[:,1]))
                ax.set_zlim(np.mean(data[:,2])-zoom*np.std(data[:,2]),np.mean(data[:,2])+zoom*np.std(data[:,2]))
            # ax.set_aspect('equal')
            ax = fig.add_subplot(322, projection='3d',azim=azim+5, elev=elev)
            ax.scatter(data[:,0], data[:,1], data[:,2],alpha=alpha,s=s,c=c,cmap='rainbow',vmin=0,vmax=np.max(c))
            # ax.set_aspect('equal')
            if zoom!=False:
                ax.set_xlim(np.mean(data[:,0])-zoom*np.std(data[:,0]),np.mean(data[:,0])+zoom*np.std(data[:,0]))
                ax.set_ylim(np.mean(data[:,1])-zoom*np.std(data[:,1]),np.mean(data[:,1])+zoom*np.std(data[:,1]))
                ax.set_zlim(np.mean(data[:,2])-zoom*np.std(data[:,2]),np.mean(data[:,2])+zoom*np.std(data[:,2]))
            plt.tight_layout()
            plt.show()
            
