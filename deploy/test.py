import argparse
import torch
from model.shufflenet import shufflenet
import os
from torchvision import transforms
from PIL import Image

label = ['acantharia_protist', 'acantharia_protist_big_center', 'acantharia_protist_halo', 'amphipods', 'appendicularian_fritillaridae', 'appendicularian_s_shape', 'appendicularian_slight_curve', 'appendicularian_straight', 'artifacts', 'artifacts_edge', 'chaetognath_non_sagitta', 'chaetognath_other', 'chaetognath_sagitta', 'chordate_type1', 'copepod_calanoid', 'copepod_calanoid_eggs', 'copepod_calanoid_eucalanus', 'copepod_calanoid_flatheads', 'copepod_calanoid_frillyAntennae', 'copepod_calanoid_large', 'copepod_calanoid_large_side_antennatucked', 'copepod_calanoid_octomoms', 'copepod_calanoid_small_longantennae', 'copepod_cyclopoid_copilia', 'copepod_cyclopoid_oithona', 'copepod_cyclopoid_oithona_eggs', 'copepod_other', 'crustacean_other', 'ctenophore_cestid', 'ctenophore_cydippid_no_tentacles', 'ctenophore_cydippid_tentacles', 'ctenophore_lobate', 'decapods', 'detritus_blob', 'detritus_filamentous', 'detritus_other', 'diatom_chain_string', 'diatom_chain_tube', 'echinoderm_larva_pluteus_brittlestar', 'echinoderm_larva_pluteus_early', 'echinoderm_larva_pluteus_typeC', 'echinoderm_larva_pluteus_urchin', 'echinoderm_larva_seastar_bipinnaria', 'echinoderm_larva_seastar_brachiolaria', 'echinoderm_seacucumber_auricularia_larva', 'echinopluteus', 'ephyra', 'euphausiids', 'euphausiids_young', 'fecal_pellet', 'fish_larvae_deep_body', 'fish_larvae_leptocephali', 'fish_larvae_medium_body', 'fish_larvae_myctophids', 'fish_larvae_thin_body', 'fish_larvae_very_thin_body', 'heteropod', 'hydromedusae_aglaura', 'hydromedusae_bell_and_tentacles', 'hydromedusae_h15', 'hydromedusae_haliscera', 'hydromedusae_haliscera_small_sideview', 'hydromedusae_liriope', 'hydromedusae_narco_dark', 'hydromedusae_narco_young', 'hydromedusae_narcomedusae', 'hydromedusae_other', 'hydromedusae_partial_dark', 'hydromedusae_shapeA', 'hydromedusae_shapeA_sideview_small', 'hydromedusae_shapeB', 'hydromedusae_sideview_big', 'hydromedusae_solmaris', 'hydromedusae_solmundella', 'hydromedusae_typeD', 'hydromedusae_typeD_bell_and_tentacles', 'hydromedusae_typeE', 'hydromedusae_typeF', 'invertebrate_larvae_other_A', 'invertebrate_larvae_other_B', 'jellies_tentacles', 'polychaete', 'protist_dark_center', 'protist_fuzzy_olive', 'protist_noctiluca', 'protist_other', 'protist_star', 'pteropod_butterfly', 'pteropod_theco_dev_seq', 'pteropod_triangle', 'radiolarian_chain', 'radiolarian_colony', 'shrimp-like_other', 'shrimp_caridean', 'shrimp_sergestidae', 'shrimp_zoea', 'siphonophore_calycophoran_abylidae', 'siphonophore_calycophoran_rocketship_adult', 'siphonophore_calycophoran_rocketship_young', 'siphonophore_calycophoran_sphaeronectes', 'siphonophore_calycophoran_sphaeronectes_stem', 'siphonophore_calycophoran_sphaeronectes_young', 'siphonophore_other_parts', 'siphonophore_partial', 'siphonophore_physonect', 'siphonophore_physonect_young', 'stomatopod', 'tornaria_acorn_worm_larvae', 'trichodesmium_bowtie', 'trichodesmium_multiple', 'trichodesmium_puff', 'trichodesmium_tuft', 'trochophore_larvae', 'tunicate_doliolid', 'tunicate_doliolid_nurse', 'tunicate_partial', 'tunicate_salp', 'tunicate_salp_chains', 'unknown_blobs_and_smudges', 'unknown_sticks', 'unknown_unclassified']
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='model checkpoints path')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
    if torch.cuda.is_available():
        device= torch.device("cuda")
#         print('\nGPU IS AVAILABLE')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    net = shufflenet().to(device)  # 修改
    
    img = Image.open(args.path).convert('RGB')
    transform = transforms.Compose([
                                  transforms.Resize((96,96)), 
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.9414, 0.9414, 0.9414],                                                           [0.1626, 0.1626, 0.1626])])
    img = transform(img)
    image = torch.unsqueeze(img,0)

    net.load_state_dict(torch.load('./93-best.pth'), device)
    net.eval()
    image = image.cuda()
    output = net(image)

#     _, pred = output.topk(1, 1, largest=True, sorted=True)
    _, pred = output.max(1)  # 返回最大值的索引
    a = pred.cpu().numpy().item()  
    print(label[a])

