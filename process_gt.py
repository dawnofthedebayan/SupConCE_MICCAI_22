import pandas as pd 
import nibabel 
import numpy as np
import uuid
cfg = {
        "smax_l": {"coronal":  { "min": {"mean":151,"std":1.899835519},
                                "max": {"mean":198.5,"std":1.414213562}
                                },

                 "sagittal":  { "min": {"mean":39.5,"std":1.322875656},
                                 "max": {"mean": 75.75,"std":1.785357107}
                                },

                 "axial":      { "min": {"mean":68.875,"std":1.964529206},
                                 "max": {"mean": 113.5,"std":1.802775638}
                                }
                 },

        "smax_r": {"coronal":  { "min": {"mean":151,"std":2.175861898},
                                "max": {"mean":198.375,"std":1.316956719}
                                },

                 "sagittal":  { "min": {"mean":95.25,"std":1.71391365},
                                 "max": {"mean": 128.875,"std":2.315032397}
                                },

                 "axial":      { "min": {"mean":66.375,"std":6.479535091},
                                 "max": {"mean": 111.5,"std":7.465145348}
                                }
                 },

        "sphen": {"coronal":  { "min": {"mean":123.75,"std":7.066647013},
                                "max": {"mean":158.375,"std":4.370036867}
                                },

                 "sagittal":  { "min": {"mean":63.625,"std":3.533323506},
                                 "max": {"mean": 103.875,"std":4.0754601}
                                },

                 "axial":      { "min": {"mean":99.625,"std":2.446298224},
                                 "max": {"mean": 127.625,"std":2.287875652}
                                }
                 },

        "sfront": {"coronal":  { "min": {"mean":185,"std":2.618614683},
                                "max": {"mean":208.2857143,"std":1.829464068}
                                },

                 "sagittal":  { "min": {"mean":54.14285714,"std":8.773801447},
                                 "max": {"mean": 109.4285714,"std":10.18201696}
                                },

                 "axial":      { "min": {"mean":126,"std":4.035556255},
                                 "max": {"mean": 156.8571429,"std":6.685347975}
                                }
                 },


        "seth": {"coronal":  { "min": {"mean":152.5714286,"std":2.258769757},
                                "max": {"mean":197.7142857,"std":4.025429372}
                                },

                 "sagittal":  { "min": {"mean":71.57142857,"std":9.897433186},
                                 "max": {"mean":101.8571429,"std":1.456862718}
                                },

                 "axial":      { "min": {"mean":104.5714286,"std":1.916629695},
                                 "max": {"mean": 129.8571429,"std":3.090472522}
                                }
                 },


        "nose": {"coronal":  { "min": {"mean":147.3333333,"std":4.229525847},
                                "max": {"mean":201.6666667,"std":2.924988129}
                                },

                 "sagittal":  { "min": {"mean":68.5,"std":1.802775638},
                                 "max": {"mean":99.33333333,"std":1.885618083}
                                },

                 "axial":      { "min": {"mean":73.16666667,"std":3.89087251},
                                 "max": {"mean": 123.8333333,"std":2.477678125}
                                }
                 },
        
      }


def __get__crop__(data,location,flip=False):
    
    #Function to crop out sub volume
    cmin,cmax =  int(location["coronal"]["min"]["mean"]),int(location["coronal"]["max"]["mean"])
    smin,smax =  int(location["sagittal"]["min"]["mean"]),int(location["sagittal"]["max"]["mean"])
    amin,amax =  int(location["axial"]["min"]["mean"]),int(location["axial"]["max"]["mean"])

    out = data[smin+10:smax-10,cmin+10:cmax-10,amin+10:amax-10]

    print(smax-smin,cmax-cmin,amax-amin)

    if flip:
        out =  np.array(np.flip(out, axis=0), dtype=np.float)
    #out = np.expand_dims(out,axis=0)
    #subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=0)
    return out



mri_root = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/all_registered_ds/{}.nii.gz"
roi_dataset =  "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/roi_dataset_m20/{}/{}.nii.gz"

df = pd.read_excel('//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/KI HCHS Project ASH 17022022.xlsx',engine='openpyxl', index_col=False)

df = df.loc[:, ['Proband','smax']]

df = df.fillna('NA')
df['Proband'] = df['Proband'].astype(str)

#print(df)
classes = {"1":"mucousa","2": "polyp","3":"Cyst"}

occurrance = {"1":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},"2":{"smax":0,"seth":0,"smax_l":0,"smax_r":0,"sfront":0,"sphen":0},"3":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},"4":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},"5":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0},"no_path":{"smax":0,"smax_l":0,"smax_r":0,"seth":0,"sfront":0,"sphen":0}}



patient_list_bad = []
patient_list_normal = []
for index, row in df.iterrows():
    
    patient_id = row["Proband"].split(" ")[0]

    if row["smax"] == "NA":

        #Normal
        try:

            img = nibabel.load(mri_root.format(patient_id.lower()))
            patient_list_normal.append(patient_id)

        except:

            print("Not found Normal",patient_id)
            continue
            
        roi = __get__crop__(img.get_data(),cfg["smax_l"])
        new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
        #nibabel.save(new_image,roi_dataset.format("good",patient_id + "_"+str(uuid.uuid1())))

        roi = __get__crop__(img.get_data(),cfg["smax_r"],flip=True)
        new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
        #nibabel.save(new_image,roi_dataset.format("good",patient_id + "_"+str(uuid.uuid1())))
        occurrance["no_path"]["smax_r"] +=  1
        occurrance["no_path"]["smax_l"] +=  1
            

    else:

        #Normal
        try:

            img = nibabel.load(mri_root.format(patient_id.lower()))
            patient_list_bad.append(patient_id)

        except:

            print("Not found Normal",patient_id)
            continue

        categories = row["smax"].split(",")
        right_bad = False
        left_bad = False

        for cat in categories:
            
            if "l" in cat:

                left_bad = True
                roi = __get__crop__(img.get_data(),cfg["smax_l"])
                new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                if "1" in cat: 
                    occurrance["1"]["smax_l"] +=  1
                if "2" in cat: 
                    occurrance["2"]["smax_l"] +=  1
                if "3" in cat: 
                    occurrance["3"]["smax_l"] +=  1

            
                #nibabel.save(new_image,roi_dataset.format("bad",patient_id + "_"+str(uuid.uuid1())))

            elif "r" in cat:
                
                right_bad  = True 
                roi = __get__crop__(img.get_data(),cfg["smax_r"],flip=True)
                new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                #nibabel.save(new_image,roi_dataset.format("bad",patient_id + "_"+str(uuid.uuid1())))
                if "1" in cat: 
                    occurrance["1"]["smax_r"] +=  1
                if "2" in cat: 
                    occurrance["2"]["smax_r"] +=  1
                if "3" in cat: 
                    occurrance["3"]["smax_r"] +=  1

            

            elif "b" in cat: 

                left_bad = True
                roi = __get__crop__(img.get_data(),cfg["smax_l"])
                new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                #nibabel.save(new_image,roi_dataset.format("bad",patient_id + "_"+str(uuid.uuid1())))

                right_bad  = True 
                roi = __get__crop__(img.get_data(),cfg["smax_r"],flip=True)
                new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                #nibabel.save(new_image,roi_dataset.format("bad",patient_id + "_"+str(uuid.uuid1())))
                if "1" in cat: 
                    occurrance["1"]["smax_l"] +=  1
                if "2" in cat: 
                    occurrance["2"]["smax_l"] +=  1
                if "3" in cat: 
                    occurrance["3"]["smax_l"] +=  1

                if "1" in cat: 
                    occurrance["1"]["smax_r"] +=  1
                if "2" in cat: 
                    occurrance["2"]["smax_r"] +=  1
                if "3" in cat: 
                    occurrance["3"]["smax_r"] +=  1

        if not right_bad:

            roi = __get__crop__(img.get_data(),cfg["smax_r"],flip=True)
            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,roi_dataset.format("good",patient_id + "_"+str(uuid.uuid1())))
            occurrance["no_path"]["smax_r"] +=  1

        if not left_bad: 

            roi = __get__crop__(img.get_data(),cfg["smax_l"])
            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,roi_dataset.format("good",patient_id + "_"+str(uuid.uuid1())))
            occurrance["no_path"]["smax_l"] +=  1


print(occurrance)
print("Normal",len(patient_list_normal))
print("Bad",len(patient_list_bad))


"""
#S MAX 
for column in df:

    
    elif column == "smax":

        df_col = df[df[column].notnull()]
        
        
            #print(row["Proband"],)
            patient_id = row["Proband"].split(" ")[0]

            try:
                img = nibabel.load(mri_root.format(patient_id.lower()))
                patient_list_normal.append(patient_id)
            except:
                print("Not found Normal",patient_id)
                continue

            roi = __get__crop__(img.get_data(),cfg["smax_l"])
            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,roi_dataset.format("nopathology","smax",patient_id + "_"+str(uuid.uuid1())))

            roi = __get__crop__(img.get_data(),cfg["smax_r"],flip=True)
            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,roi_dataset.format("nopathology","smax",patient_id + "_"+str(uuid.uuid1())))

            roi = __get__crop__(img.get_data(),cfg["seth"],flip=True)
            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,roi_dataset.format("nopathology","seth",patient_id + "_"+str(uuid.uuid1())))


            roi = __get__crop__(img.get_data(),cfg["sfront"],flip=True)
            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,roi_dataset.format("nopathology","sfront",patient_id + "_"+str(uuid.uuid1())))

            roi = __get__crop__(img.get_data(),cfg["sphen"],flip=True)
            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,roi_dataset.format("nopathology","sphen",patient_id + "_"+str(uuid.uuid1())))
                  
    else:

        
        df_col = df[df[column].notnull()]
        for index, row in df_col.iterrows():
            #print(row["Proband"],)
            class_arr  = row[column].split(",")
            patient_id = row["Proband"].split(" ")[0]
            for arr in class_arr: 
                print(column)                
                characters = list(arr)
                path_class =  characters[0]
                location   =  characters[1]

                if path_class in ["1","2","3"]: 
                    try:
                        img = nibabel.load(mri_root.format(patient_id.lower()))
                        occurrance[path_class][column] += 1
                        #patient_list_bad.append(patient_id)
                    except:
                        print("Not found Abnormal")
                        continue

                    if location == "l": 
                        #Left S max 
                        if column  == "smax":
                            roi = __get__crop__(img.get_data(),cfg["smax_l"])
                            occurrance[path_class]["smax_l"] += 1
                            patient_list_bad.append(patient_id)
                        else: 
                            roi = __get__crop__(img.get_data(),cfg[column])

                        
                        new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                        #nibabel.save(new_image,roi_dataset.format(path_class,column,patient_id + "_"+str(uuid.uuid1())))


                    elif location == "r":
                        
                        if column  == "smax":
                            occurrance[path_class]["smax_r"] += 1
                            roi = __get__crop__(img.get_data(),cfg["smax_r"],flip=True)
                            patient_list_bad.append(patient_id)
                        else: 
                            roi = __get__crop__(img.get_data(),cfg[column])

                        new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                        #nibabel.save(new_image,roi_dataset.format(path_class,column,patient_id + "_"+str(uuid.uuid1())))
                        

                    elif location == "b":

                        if column  == "smax":
                            roi_1 = __get__crop__(img.get_data(),cfg["smax_l"])
                            occurrance[path_class]["smax_l"] += 1
                            occurrance[path_class]["smax_r"] += 1
                            roi_2 = __get__crop__(img.get_data(),cfg["smax_r"],flip=True)
                            patient_list_bad.append(patient_id)

                            new_image = nibabel.Nifti1Image(roi_1.astype(np.float), affine=np.eye(4))
                            #nibabel.save(new_image,roi_dataset.format(path_class,column,patient_id + "_"+str(uuid.uuid1())))

                            new_image = nibabel.Nifti1Image(roi_2.astype(np.float), affine=np.eye(4))
                            #nibabel.save(new_image,roi_dataset.format(path_class,column,patient_id + "_"+str(uuid.uuid1())))

                        else: 

                            roi = __get__crop__(img.get_data(),cfg[column])
                            new_image = nibabel.Nifti1Image(roi.astype(np.float), affine=np.eye(4))
                            #nibabel.save(new_image,roi_dataset.format(path_class,column,patient_id + "_"+str(uuid.uuid1())))
          

                    

print(len(list(set(patient_list_normal))))
print(len(list(set(patient_list_bad))))
print(occurrance)

"""




            





