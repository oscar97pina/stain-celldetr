import torch
from ..backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .deformable_detr import DeformableDETR, PostProcess, SetCriterion
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from .matcher import build_matcher

def build_aqt_deformable_detr(cfg, backbone):
    num_classes = cfg.model.num_classes + 1
    transformer = build_deforamble_transformer(cfg)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.model.num_queries,
        num_feature_levels=cfg.model.num_feature_levels,
        aux_loss=cfg.model.aux_loss,
        with_box_refine=cfg.model.with_box_refine,
        two_stage=cfg.model.two_stage,
        backbone_align=cfg.model.uda.backbone_align,
        #level_align=cfg.model.uda.level_align if 'level_align' in cfg.model.uda else False,
        space_align=cfg.model.uda.space_align,
        channel_align=cfg.model.uda.channel_align,
        instance_align=cfg.model.uda.instance_align
    )

    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    assert cfg.matcher.name == 'HungarianMatcher', "Currently only HungarianMatcher is supported"
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.loss.class_coef, 
                   'loss_bbox': cfg.loss.bbox_coef,
                   'loss_giou': cfg.loss.giou_coef}
    
    # TODO this is a hack
    if cfg.model.aux_loss:
        aux_weight_dict = {}
        for i in range(cfg.model.transformer.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_backbone'] = cfg.loss.uda.backbone_loss_coef
    #weight_dict['loss_level_query'] = cfg.loss.uda.level_query_loss_coef if 'level_query_loss_coef' in cfg.loss.uda else 0.0
    weight_dict['loss_space_query'] = cfg.loss.uda.space_query_loss_coef
    weight_dict['loss_channel_query'] = cfg.loss.uda.channel_query_loss_coef
    weight_dict['loss_instance_query'] = cfg.loss.uda.instance_query_loss_coef

    losses = ['labels', 'boxes', 'cardinality']
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #losses += ["masks"]
    
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, 
                             focal_alpha=cfg.loss.focal_alpha, da_gamma=cfg.loss.uda.da_gamma)
    postprocessors = {'bbox': PostProcess(cfg.model.postprocess)}
    
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #postprocessors['segm'] = PostProcessSegm()
        #if args.dataset_file == "coco_panoptic":
        #    is_thing_map = {i: i <= 90 for i in range(201)}
        #    postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors