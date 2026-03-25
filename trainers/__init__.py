from .source_seg_trainer import SourceDomainTrainer
from .target_adapt_PFA_trainer import PFA_Trainer
from .target_adapt_CL_trainer import CL_Trainer
from .target_adapt_pseudo_label_trainer import PseudoLabel_Trainer

PFA_trainer = PFA_Trainer
CL_trainer = CL_Trainer