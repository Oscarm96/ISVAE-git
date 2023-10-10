import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ISVAE import *
import torch.optim as optim
from sklearn.manifold import TSNE
from src.util import *
from sklearn.preprocessing import StandardScaler




from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA





version = 1
z_dim = 2


number_f = 2
filter_w = 15**2
epochs = 5



X, X_dct, labels, n_classes, psd_dct, X_psd_class = load_data()

X_dct = torch.from_numpy(X_dct[:,  1:].astype(np.float32)) #remove continuous component of DCT
psd_dct = psd_dct[1:] #remove continuous component of DCT

plt.figure(figsize=(20, 15))
plt.plot(psd_dct[1:])
plt.show()



psd_dct_torch = torch.from_numpy(psd_dct[:].astype(np.float32)) 
batch_size = X_dct.shape[0]
x_dim = X_dct.shape[1]
training_dataset = torch.utils.data.TensorDataset(X_dct, torch.from_numpy(np.asarray(labels).astype(int)))
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size= batch_size, shuffle=False)

print('Input shape: ', X_dct.shape)
print('Number of classes: ', n_classes)



my_model = 0
f_0_epoch = []
f_ext_epoch = []
z_epoch = []
        
scores_v_epoch = []
scores_v_epoch_ext = []
scores_v_epoch_lat = []

scores_homo_epoch = []
scores_comp_epoch = []
scores_silh_epoch = []
scores_calinski_epoch = []

scores_homo_epoch_ext = []
scores_comp_epoch_ext = []
scores_silh_epoch_ext = []
scores_calinski_epoch_ext = []

scores_homo_epoch_lat = []
scores_comp_epoch_lat = []
scores_silh_epoch_lat = []
scores_calinski_epoch_lat = []
        
best_v = 0
best_e = 0
best_z = 0
best_kmeans = 0
best_enc = 0
        
best_v_ext = 0
best_e_ext = 0
best_kmeans_ext = 0
best_z_ext = 0
best_enc_ext = 0
        

best_v_lat = 0
best_e_lat = 0
best_z_lat = 0
best_kmeans_lat = 0

       

my_model = ISVAE(x_dim, z_dim, number_f, filter_w, psd_dct_torch, version)
my_model = my_model.float()

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(my_model.parameters(), lr=1e-3)
my_model.train()

total_loss = []
rec_loss = []
latent_loss = []

dic_f_0 = dict() # f_0 evolution per class and per filter along epochs
       

for txt in range(number_f):
    keys_class = list(np.arange(n_classes))
    list_class = [[] for _ in keys_class]
    dic_f_0[txt] = dict(zip(keys_class, list_class))
            

w_rec = 1/1000 
w_latent = 40

# ----- TRAINING-------------------

for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, classes = data
        inputs, classes = inputs.resize_(batch_size, x_dim), classes
        optimizer.zero_grad()

                

        if version == 2:
            new_x, x_rec, z_mean, z_sigma, z, H_list, x_filtered_list, x_filtered_sum_list, f_0_list, f_0_rec = my_model(inputs.float())
        else:
            new_x, x_rec, z_mean, z_sigma, z, H_list, x_filtered_list, x_filtered_sum_list, f_0_list= my_model(inputs.float())


        rec_l = criterion(x_rec, inputs)
        latent_l = my_model.vae.latent_loss(z_mean, z_sigma)

        rec_l = w_rec * rec_l
        latent_l = w_latent * latent_l


        f_0_epoch = f_0_epoch + [torch.cat(tuple(f_0_list), 1).clone().detach().numpy()]
        f_0_torch = torch.cat(tuple(f_0_list), 1).clone().detach().numpy()

        if version == 2:
            f_0_epoch_rec = f_0_epoch + [f_0_rec.clone().detach().numpy()]

        # ----- extended-------------------
        aux_ext = torch.cat(tuple(f_0_list), 1)
        sum_all = []

        for p in range(number_f):
            aux_sum = x_filtered_list[p]
            sum_i = torch.sum(aux_sum ** 2, 1)
            sum_i = (sum_i - torch.min(sum_i)) / (torch.max(sum_i) - torch.min(sum_i)) * x_dim
            sum_all = sum_all + [sum_i.unsqueeze(1)]

                        
        f_ext_torch = torch.cat((aux_ext, torch.cat(tuple(sum_all), 1) ),1)
        f_ext_epoch = f_ext_epoch + [f_ext_torch.clone().detach().numpy()]

        z_epoch = z_epoch + [z.clone().detach().numpy()]

        # loss
        loss = rec_l + latent_l

        loss.backward()
        optimizer.step()

        total_loss = total_loss + [float(loss.data)]

        rec_loss = rec_loss + [float(rec_l)]
        latent_loss = latent_loss + [float(latent_l)]

               
                
    # ----- write dic-------------------
    for txt in range(number_f):
        for pos, c_aux in enumerate(np.unique(classes.detach().numpy())):
            mask_f_0 = classes.detach().numpy() == c_aux
            aux = f_0_list[txt].detach().numpy()[mask_f_0]
            dic_f_0[txt][pos] = dic_f_0[txt][pos] + [np.mean(aux)]

                
    # ----- init kmeans-------------------
        
    scaler = 0
    scaler = StandardScaler()
    norm_f_0 = scaler.fit_transform(f_0_epoch[-1])
    kmeans = 0
    kmeans = KMeans(n_clusters=n_classes).fit(norm_f_0)
    scaler_ext = 0
    scaler_ext = StandardScaler()
    norm_f_0_ext = scaler_ext.fit_transform(f_ext_epoch[-1])
    kmeans_ext = 0
    kmeans_ext = KMeans(n_clusters=n_classes).fit(norm_f_0_ext)
    scaler_latent = 0
    scaler_latent = StandardScaler()
    norm_z = scaler.fit_transform(z_epoch[-1])
    kmeans_latent = 0
    kmeans_latent = KMeans(n_clusters=n_classes).fit(norm_z)
            

                
    # ----- metrics-------------------
    sc = metrics.v_measure_score(labels, kmeans.labels_)  # score
    scores_homo_epoch = scores_homo_epoch + [metrics.homogeneity_score(labels, kmeans.labels_)]
    scores_comp_epoch = scores_comp_epoch + [metrics.completeness_score(labels, kmeans.labels_)]
    scores_v_epoch = scores_v_epoch + [sc]
    scores_silh_epoch = scores_silh_epoch + [metrics.silhouette_score(norm_f_0, kmeans.labels_)]
    scores_calinski_epoch = scores_calinski_epoch + [metrics.calinski_harabasz_score(norm_f_0, kmeans.labels_)]
            
    if sc >= best_v:
        best_v = sc
        best_e = epoch
        best_kmeans = kmeans
        best_z = z
        best_enc = torch.cat(tuple(f_0_list), 1).detach().numpy()
        best_enc_aux = f_0_list
            
            
    # ----- metrics extended-------------------
    sc_ext = metrics.v_measure_score(labels, kmeans_ext.labels_)  # score ext3
    scores_homo_epoch_ext = scores_homo_epoch_ext + [metrics.homogeneity_score(labels, kmeans_ext.labels_)]
    scores_comp_epoch_ext = scores_comp_epoch_ext + [metrics.completeness_score(labels, kmeans_ext.labels_)]
    scores_v_epoch_ext = scores_v_epoch_ext + [sc_ext]
    scores_silh_epoch_ext = scores_silh_epoch_ext + [metrics.silhouette_score(norm_f_0_ext, kmeans_ext.labels_)]
    scores_calinski_epoch_ext = scores_calinski_epoch_ext + [metrics.calinski_harabasz_score(norm_f_0_ext, kmeans_ext.labels_)]
            
    if sc_ext >= best_v_ext:
        best_v_ext = sc_ext
        best_z_ext = z
        best_kmeans_ext = kmeans_ext
        best_e_ext = epoch
        best_enc_ext = torch.cat(tuple(f_0_list), 1).detach().numpy()
                
                
    # ----- metrics z-------------------
    sc_lat = metrics.v_measure_score(labels, kmeans_latent.labels_)  # score
            
    scores_homo_epoch_lat = scores_homo_epoch_lat + [metrics.homogeneity_score(labels, kmeans_latent.labels_)]
    scores_comp_epoch_lat = scores_comp_epoch_lat + [metrics.completeness_score(labels, kmeans_latent.labels_)]
    scores_v_epoch_lat = scores_v_epoch_lat + [sc]
    scores_silh_epoch_lat = scores_silh_epoch_lat + [metrics.silhouette_score(norm_z, kmeans_latent.labels_)]
    scores_calinski_epoch_lat = scores_calinski_epoch_lat + [metrics.calinski_harabasz_score(norm_z, kmeans_latent.labels_)]
            
    if sc_lat >= best_v_lat:
        best_v_lat = sc_lat
        best_e_lat = epoch
        best_z_lat = z
        best_kmeans_lat = kmeans_latent
                

           
               
            
            
    print("(Epoch %d / %d)" % (epoch, epochs))
    print("Train - REC: %.2lf;  KL: %.2lf; Total: %.2lf;  Clust score: %.2lf; Clust ext score: %.2lf; Silhouette: %.2lf; Cavinski: %.2lf" % \
        (rec_loss[-1], latent_loss[-1], total_loss[-1], sc, sc_ext,scores_silh_epoch[-1], scores_calinski_epoch[-1] ))
            
    if epoch == epochs - 1 or sc == 1:
               
        colors_real = ['#0E0097','#0089F4','#99FA56', '#FA940A', '#AB1202', '#8F02AB', '#FF3F4B', '#066804', '#81AD80','#EACE31']

            
        unique_labels = np.unique(labels)
        #------------z scatter plots-----------------
        if z_dim == 2:
                    
            plot_str = "Final latent space"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(z[:,0].detach().numpy(),z[:,1].detach().numpy(), c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$z_1$")
            plt.ylabel(r"$z_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()


                   

            plt_str = "Final latent space KMEANS"
            kmeans_label = kmeans_latent.labels_
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(z[:,0].detach().numpy(),z[:,1].detach().numpy(), c= kmeans_label,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$z_1$")
            plt.ylabel(r"$z_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()

                    
                    
            #---best latent space----
            plot_str = "Best latent space"
            b_z = best_z_lat
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(b_z[:,0].detach().numpy(),b_z[:,1].detach().numpy(), c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"Ground truth: Best v_score_lat: {best_v_lat:.2f} in epoch {best_e_lat}")
            plt.xlabel(r"$z_1$")
            plt.ylabel(r"$z_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()

             #---best latent space----
            plot_str = "Best latent space KMEANS"
            b_z = best_z_lat
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(b_z[:,0].detach().numpy(),b_z[:,1].detach().numpy(), c= best_kmeans_lat.labels_,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"KMEANS: Best v_score_lat: {best_v_lat:.2f} in epoch {best_e_lat}")
            plt.xlabel(r"$z_1$")
            plt.ylabel(r"$z_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    
                    
        else:
                    
            plot_str = "Final latent space"
            tsne_features = TSNE(n_components=2).fit_transform(z.detach().numpy())
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$tsne_1$ ($z$)")
            plt.ylabel(r"$tsne_2$ ($z$)")
            plt.ticklabel_format(useOffset=False)
            plt.show()

                    
            

            plt_str = "Final latent space KMEANS"
            kmeans_label = kmeans_latent.labels_
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= kmeans_label,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$tsne_1$ ($z$)")
            plt.ylabel(r"$tsne_2$ ($z$)")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    
             #---best latent space----

            tsne_features = TSNE(n_components=2).fit_transform(best_z_lat.detach().numpy())
            plot_str = "Best latent space best"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"Ground truth: Best v_score_lat: {best_v_lat:.2f} in epoch {best_e_lat}")
            plt.xlabel(r"$tsne_1$ ($z$)")
            plt.ylabel(r"$tsne_2$ ($z$)")
            plt.ticklabel_format(useOffset=False)
            plt.show()

            plot_str = "Best latent space KMEANS"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= best_kmeans_lat.labels_,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"KMEANS: Best v_score_lat: {best_v_lat:.2f} in epoch {best_e_lat}")
            plt.xlabel(r"$z_1$")
            plt.ylabel(r"$z_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()

                    
        #------------f_0 scatter plots-----------------
               
        if number_f == 2:
            plot_str = "f_0"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(f_0_torch[:,0],f_0_torch[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    



            kmeans_label = kmeans.labels_
            plot_str = "f_0 KMEANS"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(f_0_torch[:,0],f_0_torch[:,1], c= kmeans_label,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    
                    
            #---best f_0----
            b_f_0 = best_enc
            plot_str = "Best f_0"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(b_f_0[:,0],b_f_0[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"Ground truth: Best v_score: {best_v:.2f} in epoch {best_e}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    

                    
            plot_str = "Best f_0 KMEANS"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(b_f_0[:,0],b_f_0[:,1], c=  best_kmeans.labels_,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(f"KMEANS: Best v_score: {best_v:.2f} in epoch {best_e}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                                
                    
                   
            #---best f_enc_ext----

            b_f_0 = best_enc_ext
            plot_str = "Best f_0"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(b_f_0[:,0],b_f_0[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"Ground truth: Best v_score: {best_v_ext:.2f} in epoch {best_e_ext}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    

                    
            plot_str = "Best f_0 KMEANS"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(b_f_0[:,0],b_f_0[:,1], c= best_kmeans_ext.labels_,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(f"KMEANS: Best v_score: {best_v_ext:.2f} in epoch {best_e_ext}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    
                    
        else:

            tsne_features = 0
            tsne_features = TSNE(n_components=2).fit_transform(f_0_torch)


            plot_str = "f_0"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    



            kmeans_label = kmeans.labels_
            plot_str = "f_0 KMEANS"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= kmeans_label,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(plot_str)
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()

                    
            #---best f_enc----
            tsne_features = 0
            tsne_features = TSNE(n_components=2).fit_transform(best_enc)


            plot_str = "Best f_0"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"Ground truth: Best v_score: {best_v:.2f} in epoch {best_e}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    

                    
            plot_str = "Best f_0 KMEANS"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c=  best_kmeans.labels_,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(f"KMEANS: Best v_score: {best_v:.2f} in epoch {best_e}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()



                   
                   
            #---best f_enc_ext----
            tsne_features = 0
            tsne_features = TSNE(n_components=2).fit_transform(best_enc_ext)

            plot_str = "Best f_0"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= labels,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid()
            plt.title(f"Ground truth: Best v_score: {best_v_ext:.2f} in epoch {best_e_ext}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    

                    
            plot_str = "Best f_0 KMEANS"
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_features[:,0],tsne_features[:,1], c= best_kmeans_ext.labels_,  marker='o', edgecolor='none', cmap=plt.cm.get_cmap('turbo', 10), s=10)
            plt.grid()
            plt.title(f"KMEANS: Best v_score: {best_v_ext:.2f} in epoch {best_e_ext}")
            plt.xlabel(r"$f_1$")
            plt.ylabel(r"$f_2$")
            plt.ticklabel_format(useOffset=False)
            plt.show()
                    
        colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'y', 'r', 'b', 'g', 'm', 'c', 'y', 'k', 'y']
        # generic histogram 
        plt.figure(figsize=(15, 10))
        plt.title('Global PSD and histograms of filters')
        plt.subplot(number_f + 1, 1, 1)
        plt.plot(psd_dct[1:])
        for j in range(number_f):
            plt.subplot(number_f + 1, 1, j+2)
            plt.hist(best_enc_aux[j].detach().numpy(), x_dim, density=True, facecolor=colors[j], alpha=0.5, range=(0, x_dim))
            str_f = r"$f_"+str(j+1)+"$"
            plt.xlabel(str_f)
        plt.show()


        # histograma por clase
        for pos, j in enumerate(np.unique(classes.detach().numpy())):
            plot_str = "histogram_class_"+str(j)+"_"
            plt.figure(figsize=(20, 15))
            plt.title('Histogram for class ' + str(j))
            mask = classes.detach().numpy() == j
            inputs_masked_hist = inputs.detach().numpy()[mask, :]
            plt.subplot(number_f + 2, 1, 1)
            plt.title('Class psd from class: ' + str(j))
            plt.plot(X_psd_class[pos, :])
            for t in range(number_f):
                plt.subplot(number_f + 2, 1, t + 2)
                f_0_masked = best_enc_aux[t].detach().numpy()[mask]
                plt.hist(f_0_masked, x_dim, density=True, facecolor=colors[t], alpha=0.5, range=(0, x_dim))
                mean_f_0 = np.mean(f_0_masked)
            plt.subplot(number_f + 2, 1, number_f + 2)
            for t in range(number_f):
                f_0_masked = best_enc_aux[t].detach().numpy()[mask]
                plt.hist(f_0_masked, x_dim, density=True, facecolor=colors[t], alpha=0.5, range=(0, x_dim))
                mean_f_0 = np.mean(f_0_masked)
            plt.show()

        final_epoch = epoch

                
    


plt.figure(figsize=(20, 10))
plt.subplot(3, 1, 1)
plt.title('Basic configuration: V_score (homo. and compl.) ' + 'Best v_score: ' + str(best_v))
plt.plot(scores_v_epoch, label='v_score', c='k')
plt.axvline(best_e, c='k')
plt.plot(scores_homo_epoch, label='homogeneity', c='b')
plt.plot(scores_comp_epoch, label='completeness', c='r')
plt.legend()
plt.subplot(3, 1, 2)
plt.title('Silhouette Coefficient')
plt.plot(scores_silh_epoch, c='m')
plt.subplot(3, 1, 3)
plt.title('Calinski-Harabasz Index')
plt.plot(scores_calinski_epoch,  c='g')
plt.show()



plt.figure(figsize=(20, 10))
plt.subplot(3, 1, 1)
plt.title('Extended configuration:V_score (homo. and compl.) ' + 'Best v_score: ' + str(best_v_ext))
plt.plot(scores_v_epoch_ext, label='v_score', c='k')
plt.axvline(best_e_ext, c='k')
plt.plot(scores_homo_epoch_ext, label='homogeneity', c='b')
plt.plot(scores_comp_epoch_ext, label='completeness', c='r')
plt.legend()
plt.subplot(3, 1, 2)
plt.title('Silhouette Coefficient')
plt.plot(scores_silh_epoch_ext, c='m')
plt.axvline(best_e_ext, c='k')
plt.subplot(3, 1, 3)
plt.title('Calinski-Harabasz Index')
plt.plot(scores_calinski_epoch_ext, c='g')
plt.axvline(best_e_ext, c='k')
plt.show()

f_0_evo = np.zeros(((number_f, final_epoch+1)))
f_sum_evo = np.zeros(((number_f, final_epoch+1)))




plot_str = "summary_"
plt.figure(figsize=(30, 20))
plt.subplot(4, 1, 1)
plt.title('Reconstruction')
plt.plot(rec_loss, c='r')
plt.subplot(4, 1, 2)
plt.title('KL')
plt.plot(latent_loss, c='r')
plt.subplot(4, 1, 3)
plt.title('score_clust', c='k')
plt.plot(scores_v_epoch, label='v_score', c='k')
plt.plot(scores_homo_epoch, label='homogeneity', c='b')
plt.plot(scores_comp_epoch, label='completeness', c='r')
plt.legend()
plt.subplot(4, 1, 4)
plt.title('f_0 evolution total')
for n_f in range(number_f):
    aux = np.zeros((n_classes, np.asarray(dic_f_0[n_f][0]).shape[0]))
    for n_c in range(n_classes):
        aux[n_c, :] = np.asarray(dic_f_0[n_f][n_c])
    plt.plot(np.mean(aux, axis=0), label='f_0 in F' + str(n_f), c=colors[n_f])
    f_0_evo[n_f, :] = np.mean(aux, axis=0)
            
    plt.ylim([-20, my_model.x_dim +20])
    plt.legend()
plt.show()

      
plot_str = 'evo_f_0'
plt.figure(figsize=(25, 20))
plt.subplot(number_f + 2, 1, 1)
plt.title('score_clust')
plt.plot(scores_v_epoch, label='v_score', c='k')
plt.plot(scores_homo_epoch, label='homogeneity', c='b')
plt.plot(scores_comp_epoch, label='completeness', c='r')
plt.legend()
plt.subplot(number_f + 2, 1, 2)
plt.title('Evolution f_0 Filters')
for txt in range(number_f):
    plt.plot(f_0_evo[txt, :], label='F' + str(txt), c=colors[txt])
    plt.legend()
plt.ylim([-50, my_model.x_dim+50])
for txt in range(number_f):
    plt.subplot(number_f + 2, 1, txt + 3)
    plt.title('Evolution f_0 (Filter ' + str(txt) + ') per class')
    for pos in range(n_classes):
        plt.plot(dic_f_0[txt][pos], label='C' + str(pos), c=colors[pos])
        plt.legend()
    plt.ylim([-50, my_model.x_dim+50])
    str_f = r"$f_"+str(txt+1)+"$"
    plt.xlabel(str_f)
plt.show()
