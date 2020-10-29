
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import shutil
import os, sys, glob
import smplx
import smplx.joint_names
import argparse
import gc

from tqdm import tqdm
from tools.objectmodel import ObjectModel
from tools.cfg_parser import Config
from tools.utils import makepath, makelogger
from tools.meshviewer import Mesh
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import to_cpu
import tools.consts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_grab_vertices(cfg, logger=None, **params):

    grab_path = cfg.grab_path
    out_path = cfg.out_path
    makepath(out_path)

    if logger is None:
        logger = makelogger(log_dir=os.path.join(out_path, 'grab_preprocessing.log'), mode='a').info
    else:
        logger = logger
    logger('Starting to get vertices for GRAB!')

        
    all_seqs = glob.glob(grab_path + '/*/*.npz')
    

    logger('Total sequences: %d' % len(all_seqs))

    # stime = datetime.now().replace(microsecond=0)
    # shutil.copy2(sys.argv[0],
    #              os.path.join(out_path,
    #                           os.path.basename(sys.argv[0]).replace('.py','_%s.py' % datetime.strftime(stime,'%Y%m%d_%H%M'))))


    if out_path is None:
        out_path = grab_path

    for sequence in tqdm(all_seqs):

        gc.collect()

        outfname = makepath(sequence.replace(grab_path,out_path).replace('.npz', '_verts_body.npz'), isfile=True)

        action_name = os.path.basename(sequence)
        #remove this
        if os.path.exists(outfname.replace('_verts_body.npz', '_verts_object.npz')):
            logger('Results for %s split already exist.' % (action_name))
            continue
        else:
            logger('Processing data for %s split.' % (action_name))

        seq_data = parse_npz(sequence)
        n_comps = seq_data['n_comps']
        gender = seq_data['gender']

        T = seq_data.n_frames

        if cfg.save_body_verts:

            sbj_mesh = os.path.join(grab_path, '..', seq_data.body.vtemp)
            sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

            sbj_m = smplx.create( model_path=cfg.model_path,
                                  model_type='smplx',
                                  gender=gender,
                                  num_pca_comps=n_comps,
                                  v_template = sbj_vtemp,
                                  batch_size=T)

            sbj_parms = params2torch(seq_data.body.params)
            output_sbj = sbj_m(**sbj_parms, no_grad=True)
            verts_sbj = to_cpu(output_sbj.vertices)
            joints_sbj = to_cpu(output_sbj.joints) # to get the body joints (it includes some additional landmarks, check smplx repo.
            np.savez_compressed(outfname, verts_body=verts_sbj)
        
        if cfg.save_hand_joints:
            if not cfg.save_body_verts:
                sbj_mesh = os.path.join(grab_path, '..', seq_data.body.vtemp)
                sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

                sbj_m = smplx.create( model_path=cfg.model_path,
                                    model_type='smplx',
                                    gender=gender,
                                    num_pca_comps=n_comps,
                                    v_template = sbj_vtemp,
                                    batch_size=T)

                sbj_parms = params2torch(seq_data.body.params)
                output_sbj = sbj_m(**sbj_parms)
                verts_sbj = to_cpu(output_sbj.vertices)
                joints_sbj = to_cpu(output_sbj.joints) # to get the body joints (it includes some additional landmarks, check smplx repo.
                
            lhand_parms = params2torch(seq_data.lhand.params)
            rhand_parms = params2torch(seq_data.rhand.params)
            joint_names = smplx.joint_names.JOINT_NAMES
            rhand_joints = output_sbj.joints[:, [joint_names.index(name) for name in tools.consts.RHAND_JOINT_NAMES], :].detach().numpy()
            lhand_joints = output_sbj.joints[:, [joint_names.index(name) for name in tools.consts.LHAND_JOINT_NAMES], :].detach().numpy()
            smplx_vertex_ids = smplx.vertex_ids.vertex_ids['smplx']
            rhand_tips = output_sbj.vertices[:,[smplx_vertex_ids[name] for name in tools.consts.RHAND_VERTEX_TIPS],:].detach().numpy()
            lhand_tips = output_sbj.vertices[:,[smplx_vertex_ids[name] for name in tools.consts.LHAND_VERTEX_TIPS],:].detach().numpy()
            #lhand_all = torch.cat((lhand_joints, lhand_tips), axis=1)
            #rhand_all = torch.cat((rhand_joints, rhand_tips), axis=1)
            lhand_transl = lhand_parms['transl'].detach().numpy()
            rhand_transl = rhand_parms['transl'].detach().numpy()
            
            lhand_orient = -lhand_parms['global_orient']
            rhand_orient = -rhand_parms['global_orient']
            lhand_rot_mats = to_cpu(smplx.lbs.batch_rodrigues(lhand_orient.view(-1, 3)).view([np.shape(lhand_joints)[0], 3, 3]))
            rhand_rot_mats = to_cpu(smplx.lbs.batch_rodrigues(lhand_orient.view(-1, 3)).view([np.shape(rhand_joints)[0], 3, 3]))
            
                      
            np.savez_compressed(outfname.replace('_verts_body.npz', '_hand_joints.npz'), lhand_tips=lhand_tips, rhand_tips = rhand_tips, 
                                lhand_joints=lhand_joints, rhand_joints=rhand_joints,
                                lhand_trans = lhand_transl, rhand_trans=rhand_transl, 
                                lhand_rot = lhand_rot_mats, rhand_rot = rhand_rot_mats)
        
        if cfg.save_contact:
            #object vertices, so need that
            object_contact = seq_data['contact']['object']
            body_contact = seq_data['contact']['body']
            frame_mask = (seq_data['contact']['object']>0).any(axis=1)
            np.savez_compressed(outfname.replace('_verts_body.npz', '_contact_info.npz'), object_contact_mask = frame_mask,
                                object_contact = object_contact, body_contact = body_contact)

        if cfg.save_lhand_verts:
            lh_mesh = os.path.join(grab_path, '..', seq_data.lhand.vtemp)
            lh_vtemp = np.array(Mesh(filename=lh_mesh).vertices)

            lh_m = smplx.create(model_path=cfg.model_path,
                                model_type='mano',
                                is_rhand=False,
                                v_template = lh_vtemp,
                                num_pca_comps=n_comps,
                                flat_hand_mean = True,
                                batch_size=T)

            lh_parms = params2torch(seq_data.lhand.params)
            lh_output = lh_m(**lh_parms)
            verts_lh = to_cpu(lh_output.vertices)
            joints_lh = to_cpu(lh_output.joints) # to get the hand joints          
            np.savez_compressed(outfname.replace('_verts_body.npz', '_verts_lhand.npz'), verts_body=verts_lh)

        if cfg.save_rhand_verts:
            rh_mesh = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
            rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)

            rh_m = smplx.create(model_path=cfg.model_path,
                                model_type='mano',
                                is_rhand = True,
                                v_template = rh_vtemp,
                                num_pca_comps=n_comps,
                                flat_hand_mean=True,
                                batch_size=T)

            rh_parms = params2torch(seq_data.rhand.params)
            rh_output = rh_m(**rh_parms)
            verts_rh = to_cpu(rh_output.vertices)
            joints_rh = to_cpu(rh_output.joints) # to get the hand joints
            np.savez_compressed(outfname.replace('_verts_body.npz', '_verts_rhand.npz'), verts_body=verts_rh)


        if cfg.save_object_verts:

            obj_mesh = os.path.join(grab_path, '..', seq_data.object.object_mesh)
            obj_vtemp = np.array(Mesh(filename=obj_mesh).vertices)
            sample_id = np.random.choice(obj_vtemp.shape[0], cfg.n_verts_sample, replace=False)
            obj_m = ObjectModel(v_template=obj_vtemp[sample_id],
                                batch_size=T)
            obj_parms = params2torch(seq_data.object.params)
            verts_obj = to_cpu(obj_m(**obj_parms).vertices)
            np.savez_compressed(outfname.replace('_verts_body.npz', '_verts_object.npz'), verts_object=verts_obj)



        logger('Processing finished')


if __name__ == '__main__':

    msg = ''' 
        This code will process the grab dataset and save the desired vertices for body, hands, and object.
        
        Please do the following steps before starting the GRAB dataset processing:
        1. Download GRAB dataset from the website https://grab.is.tue.mpg.de/ 
        2. Set the grab_path, out_path to the correct folder
        3. Change the configuration file for your desired vertices
        4. In case you need body or hand vertices make sure to set the model_path
            to the models downloaded from smplx website
            '''

    parser = argparse.ArgumentParser(description='GRAB-vertices')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--out-path', default=None, type=str,
                        help='The path to the folder to save the vertices')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')


    args = parser.parse_args()

    grab_path = args.grab_path
    out_path = args.out_path
    model_path = args.model_path

    if out_path is None:
        out_path = grab_path

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # out_path = 'PATH_TO_THE LOCATION_TO_SAVE_DATA/grab'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {

            'save_body_verts': True, # if True, will compute and save the body vertices in the specified path
            'save_object_verts': True,
            'save_lhand_verts': False,
            'save_rhand_verts': False,
            'save_hand_joints': False,

            # number of vertices samples for each object
            'n_verts_sample': 1024,

            #IO path
            'grab_path': grab_path,
            'out_path': out_path,

            # body and hand model path
            'model_path':model_path,
        }

    log_dir = os.path.join(out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info
    logger(msg)

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, './configs/get_vertices_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path)
    cfg.write_cfg(write_path=cfg.out_path + '/get_vertices_cfg.yaml')

    save_grab_vertices(cfg, logger)

