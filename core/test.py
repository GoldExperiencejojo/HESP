import numpy as np
import torch

from core import evaluation

def test(prompt, normalization, criterion, testloader, outloader, epoch=None, **options):
    correct_a, correct_c, correct_e, total = 0, 0, 0, 0
    torch.cuda.empty_cache()
    all_correct, all_total = 0, 0
    _pred_k_a, _pred_u_a, _labels = [], [], []
    _pred_k_c, _pred_u_c = [], []
    _pred_k_e, _pred_u_e = [], []

    with torch.no_grad():
        unknown_p_v = criterion.unknown_p_v
        unknown_p_v_features = prompt.image_encoder(unknown_p_v)

        my_center = unknown_p_v_features
        for frames, labels, label_text, video_name in testloader:
            b,t,c,h,w = frames.size()
            if options['use_gpu']:
                frames, labels = frames.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                noise = prompt.perturbation.cuda()
                noise = noise.repeat(frames.size(0), 1, 1, 1)

                row, col = options['patch_position'][0], options['patch_position'][1]
                patch_size = options['patch_size']
                half_patch_size = patch_size // 2
                if row > 223 - half_patch_size:
                    row = 223 - half_patch_size
                if col > 223 - half_patch_size:
                    col = 223 - half_patch_size
                if row < half_patch_size:
                    row = half_patch_size
                if col < half_patch_size:
                    col = half_patch_size
                for i in range(8):
                    frames[:, i, :, row-half_patch_size:row+half_patch_size, col-half_patch_size:col+half_patch_size] = frames[:, i, :, row-half_patch_size:row+half_patch_size, col-half_patch_size:col+half_patch_size] + noise                       

                frames = normalization(frames)
                
                frames = frames.view(-1,c,h,w)
                clip_logits, visual_embedding = prompt(frames,b,t)

                visual_embedding = visual_embedding.float()
                y = prompt.fc(visual_embedding)
                logits, _ = criterion(visual_embedding, y, my_center)

                predictions_a = logits.data.max(1)[1]
                total += labels.size(0)
                correct_a += (predictions_a == labels.data).sum()
                _pred_k_a.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())
                
                predictions_c = clip_logits.data.max(1)[1]
                correct_c += (predictions_c == labels.data).sum()
                _pred_k_c.append(clip_logits.data.cpu().numpy())
                
                votes = torch.stack([clip_logits, logits])
                e_logits = votes.mean(dim=0)
                predictions_e = torch.argmax(votes.mean(dim=0), dim=1)
                correct_e += (predictions_e == labels.data).sum()
                _pred_k_e.append(e_logits.data.cpu().numpy())
                e_pro = torch.softmax(e_logits, dim=-1)
                max_epro, max_epro_index = e_pro.data.max(1)

        for batch_idx, (frames, labels, label_text, video_name) in enumerate(outloader):
            all_total += labels.size(0)
            b,t,c,h,w = frames.size()
            if options['use_gpu']:
                frames, labels = frames.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                noise = prompt.perturbation.cuda()
                noise = noise.repeat(frames.size(0), 1, 1, 1)

                row, col = options['patch_position'][0], options['patch_position'][1]
                patch_size = options['patch_size']
                half_patch_size = patch_size // 2
                if row > 223 - half_patch_size:
                    row = 223 - half_patch_size
                if col > 223 - half_patch_size:
                    col = 223 - half_patch_size
                if row < half_patch_size:
                    row = half_patch_size
                if col < half_patch_size:
                    col = half_patch_size
                for i in range(8):
                    frames[:, i, :, row-half_patch_size:row+half_patch_size, col-half_patch_size:col+half_patch_size] = frames[:, i, :, row-half_patch_size:row+half_patch_size, col-half_patch_size:col+half_patch_size] + noise  
                        
                frames = normalization(frames)
                frames = frames.view(-1,c,h,w)
                clip_logits, visual_embedding = prompt(frames,b,t)
                
                visual_embedding = visual_embedding.float()   
                y = prompt.fc(visual_embedding)
                logits, _ = criterion(visual_embedding, y, my_center)
                
                _pred_u_c.append(clip_logits.data.cpu().numpy())
                _pred_u_a.append(logits.data.cpu().numpy())
                 
                votes = torch.stack([clip_logits, logits])
                e_logits = votes.mean(dim=0)
                _pred_u_e.append(e_logits.data.cpu().numpy())
                
                
    # Accuracy
    _labels = np.concatenate(_labels, 0)
    
    acc = float(correct_e) * 100. / float(total)

    _pred_k_e = np.concatenate(_pred_k_e, 0)
    _pred_u_e = np.concatenate(_pred_u_e, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k_e, axis=1), np.max(_pred_u_e, axis=1)
    results_e = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k_e, _pred_u_e, _labels)

    results_e['ACC_CLOSE'] = acc
    results_e['OSCR'] = _oscr_socre * 100.
    
    return results_e