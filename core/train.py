import torch
from utils import AverageMeter
import clip
import numpy as np
import cv2

def returnCAM(feature_conv, weight_softmax, class_idx):
    b, c, h, w = feature_conv.shape  
    output_cam = []
    for idx in class_idx: 
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))  
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min()) 
        cam_img = np.uint8(255 * cam_img)  
        output_cam.append(cam_img)
    return output_cam

def train(net0, preprocess, prompt, normalization, criterion, prompt_CEloss, contrastive_loss, optimizer, trainloader, epoch=None, **options):
    losses = AverageMeter()
    torch.cuda.empty_cache()

    logits_list = []
    loss_all = 0
    flag_attention = True
    for batch_idx, (frames, labels, label_text, video_name) in enumerate(trainloader):
        b,t,c,h,w = frames.size()
        if options['use_gpu']:
            frames, labels = frames.cuda(), labels.cuda()

        if epoch == 0 and flag_attention:
            flag_attention = False
            fc0 = torch.nn.Linear(768, options['num_classes']).cuda()
            
            img = frames[0,0,:,:,:].unsqueeze(0)
            fc_weights = fc0.weight.detach().cpu().numpy()
            net0.eval()
            visual_embedding, features = net0.encode_image(img, True)
            label_text = clip.tokenize(options['classes_names']).cuda()
            text_embedding = net0.encode_text(label_text)

            x_visual = visual_embedding / visual_embedding.norm(dim=-1, keepdim=True)
            x_text = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            logit_scale = net0.logit_scale.exp()
            clip_logits = logit_scale * x_visual @ x_text.t()
            output = clip_logits
            features = features.detach().cpu().numpy() 
            pred = torch.nn.functional.softmax(output, dim=1).data.squeeze()
            probs, idx = pred.sort(0, True)    
            probs = probs.cpu().numpy() 
            idx = idx.cpu().numpy() 
            CAMs = returnCAM(features, fc_weights, [idx[0]])

            img = frames[0,0,:,:,:].cpu().numpy().transpose(1,2,0)
            img = np.uint8(255 * img)
            height, width, _ = img.shape  
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET) 
            result = heatmap * 0.3 + img
            output_path = 'attention/new/' + options['ids'] +'.jpg'
            cv2.imwrite(output_path, result)

            hsv_image = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

            # 定义红色的范围
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])

            # 创建红色掩码
            mask = cv2.inRange(hsv_image, lower_red, upper_red)
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 找到最大的轮廓
            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            # 计算最大轮廓的中心点
            M = cv2.moments(max_contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # 输出结果
            # print("最红的点的坐标：", (cx, cy))
            options['patch_position'] = [cy, cx]             

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            
            noise = prompt.perturbation.cuda()
            noise = noise.repeat(frames.size(0), 1, 1, 1)
            noise.retain_grad()
            
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
            frames.require_grad = True
            
            frames = frames.view(-1,c,h,w)
            clip_logits, visual_embedding = prompt(frames,b,t)
            clip_loss_k = prompt_CEloss(clip_logits, labels)
            
            clip_pro = torch.softmax(clip_logits, dim=-1)
            max_c_pro = clip_pro.data.max(1)[0]

            unclass_text_prompt = "This video is not "
            # 在每个单词前添加prompt
            unclass_text = [unclass_text_prompt + word + "." for word in options['classes_names']]
            unclass_text_token = clip.tokenize(unclass_text).cuda()
            unknown_p_t_features = prompt.text_encoder(unclass_text_token)
            unknown_p_v = criterion.unknown_p_v
            unknown_p_v_features = prompt.image_encoder(unknown_p_v)
            
            unclass_image_features = unknown_p_v_features / unknown_p_v_features.norm(dim=-1, keepdim=True)
            unclass_text_features = unknown_p_t_features / unknown_p_t_features.norm(dim=-1, keepdim=True)

            logit_scale = prompt.logit_scale.exp()
            unclass_prompt_logits = logit_scale * unclass_image_features @ unclass_text_features.t()
            class_label = list(range(options['num_classes']))
            class_label = torch.tensor(class_label).cuda()
            clip_loss_unclass = prompt_CEloss(unclass_prompt_logits, class_label)
            
            my_center = unknown_p_v_features

            c_loss = contrastive_loss(visual_embedding, labels)
                    
            visual_embedding = visual_embedding.float()
            y = prompt.fc(visual_embedding)
            logits, osr_loss = criterion(visual_embedding, y, my_center, labels)
            a_pro = torch.softmax(logits, dim=-1)
            max_a_pro = a_pro.data.max(1)[0]
                        
            votes = torch.stack([clip_logits, logits])
            e_pred = torch.argmax(votes.mean(dim=0), dim=1)
            loss_e = prompt_CEloss(votes.mean(dim=0), labels)
            
            votes_pro = torch.stack([clip_pro, a_pro])
            max_e_pro = votes_pro.mean(dim=0).data.max(1)[0]
            logits_list.append(max_e_pro)

            loss = clip_loss_k + c_loss + osr_loss + clip_loss_unclass + loss_e
            loss.backward()
            
            # update the perturbation
            grad_p_u = unknown_p_v.grad
            grad_p_u = grad_p_u.mean(0).squeeze(0)
            g_norm_u = torch.norm(grad_p_u.view(-1), dim=0).view(1, 1, 1)
            scaled_g_u = grad_p_u / (g_norm_u + 1e-10)
            updated_u = scaled_g_u * 50 
            criterion.unknown_p_v.data = criterion.unknown_p_v.data - updated_u

            grad_p_t = noise.grad
            grad_p_t = grad_p_t.mean(0).squeeze(0)
            g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
            scaled_g = grad_p_t / (g_norm + 1e-10)
            updated_pad = scaled_g * 50 
            prompt.perturbation.data = prompt.perturbation.data - updated_pad.detach().cpu()
            prompt.zero_grad()
            
            optimizer.step()
        
        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
            print('clip_loss_k: {}\tc_loss: {}\tosr_loss: {}\tclip_loss_unclass: {}\te_loss: {}'.format(clip_loss_k.item(), c_loss.item(), osr_loss.item(), clip_loss_unclass.item(), loss_e.item()))
            options['writer'].add_scalar('train_loss', losses.val, epoch*len(trainloader)+batch_idx)
        loss_all += losses.avg

    if epoch == 0:
        return loss_all, logits_list, [cy, cx]
    else:
        return loss_all, logits_list
