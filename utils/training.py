import torch
import torch.nn as nn
import gc
import logging
from tqdm import tqdm
from .metrics import MetricsCalculator
from .helpers import save_checkpoint
from data.transforms import get_transforms, AdaptiveAugmentation


class AblationTrainer:
    """Simplified trainer for ablation studies"""

    def __init__(self, model, appearance_model, depth_model, motion_model,
                 train_loader, val_loader, config, device,
                 use_appearance=True, use_motion=True, use_depth=True):
        self.model = model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_appearance = use_appearance
        self.use_motion = use_motion
        self.use_depth = use_depth

        # Freeze backbones
        for m in [appearance_model, depth_model, motion_model]:
            for param in m.parameters():
                param.requires_grad = False

        # Optimizer for trainable parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['epochs']
        )

        self.best_val_acc = 0
        self.metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train(self):
        """Train the ablation model"""
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping']['patience']
        no_improve = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_preds, train_labels = [], []

            for frames, labels, _ in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                frames = frames.to(self.device)
                labels = labels.to(self.device).float()

                with torch.no_grad():
                    app_feats = self.appearance_model(frames) if self.use_appearance else None
                    depth_feats = self.depth_model(frames) if self.use_depth else None
                    motion_feats = self.motion_model(frames) if self.use_motion else None

                    # Handle None values with appropriate dimensions
                    if app_feats is None:
                        app_feats = torch.zeros(frames.shape[0], 16, 512).to(self.device)
                    if depth_feats is None:
                        depth_feats = torch.zeros(frames.shape[0], 128).to(self.device)
                    if motion_feats is None:
                        motion_feats = torch.zeros(frames.shape[0], 768).to(self.device)

                outputs = self.model(app_feats, depth_feats, motion_feats)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Validation
            val_loss, val_acc, val_f1 = self._validate()

            # Update metrics
            train_acc = accuracy_score(train_labels, train_preds)
            self.metrics['train_loss'].append(train_loss / len(self.train_loader))
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                no_improve = 0
                torch.save(self.model.state_dict(), f"best_ablation_model.pth")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(self.train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Load best model
        self.model.load_state_dict(torch.load("best_ablation_model.pth"))
        return self.model, self.metrics

    def _validate(self):
        """Validation step"""
        self.model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for frames, labels, _ in self.val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device).float()

                app_feats = self.appearance_model(frames) if self.use_appearance else None
                depth_feats = self.depth_model(frames) if self.use_depth else None
                motion_feats = self.motion_model(frames) if self.use_motion else None

                if app_feats is None:
                    app_feats = torch.zeros(frames.shape[0], 16, 512).to(self.device)
                if depth_feats is None:
                    depth_feats = torch.zeros(frames.shape[0], 128).to(self.device)
                if motion_feats is None:
                    motion_feats = torch.zeros(frames.shape[0], 768).to(self.device)

                outputs = self.model(app_feats, depth_feats, motion_feats)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        return val_loss / len(self.val_loader), val_acc, val_f1
class Trainer:
    """Main trainer class with frozen backbones and CMAD support"""

    def __init__(self, model, appearance_model, depth_model, motion_model,
                 train_loader, val_loader, config, device, cmad_enabled=True):
        self.model = model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.cmad_enabled = cmad_enabled

        # Ensure backbones are frozen
        self._freeze_backbones()

        # Only optimize trainable parameters (projection layers, transformer, cross-attention, classifier)
        self.trainable_params = self._get_trainable_params()

        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        self.metrics_calc = MetricsCalculator()

        self.best_val_acc = 0
        self.best_epoch = 0
        self.no_improve = 0
        self.metrics = self._init_metrics()

        # Log trainable parameters
        total_params = sum(p.numel() for p in self.trainable_params)
        print(f"\nTrainable parameters: {total_params:,}")

    def _freeze_backbones(self):
        """Freeze all pre-trained backbone models"""
        # Freeze appearance model (CLIP)
        for param in self.appearance_model.parameters():
            param.requires_grad = False

        # Freeze depth model (MiDaS)
        for param in self.depth_model.parameters():
            param.requires_grad = False

        # Freeze motion model (VideoMAE)
        for param in self.motion_model.parameters():
            param.requires_grad = False

        print("All backbone models frozen (CLIP, MiDaS, VideoMAE)")

    def _get_trainable_params(self):
        """Get only trainable parameters (projection layers, transformer, cross-attention, classifier)"""
        trainable_params = []

        # Trainable components in fusion model
        for name, param in self.model.named_parameters():
            # These are the components we want to train
            trainable_components = [
                'app_proj', 'motion_proj', 'depth_proj',  # Projection layers
                'temporal_model',  # Transformer encoder
                'app_motion_attn', 'app_depth_attn',  # Cross-attention modules
                'classifier',  # MLP classifier
                'cmad_projection'  # CMAD projection (if exists)
            ]

            if any(component in name for component in trainable_components):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        return trainable_params

    def _get_criterion(self):
        from .metrics import FocalLoss
        return FocalLoss(
            alpha=self.config['training']['focal_alpha'],
            gamma=self.config['training']['focal_gamma']
        )

    def _get_optimizer(self):
        return torch.optim.AdamW(
            self.trainable_params,
            lr=self.config['training']['learning_rate'],  # 1e-5 as specified
            weight_decay=self.config['training']['weight_decay']
        )

    def _get_scheduler(self):
        if self.config['training']['scheduler']['type'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],  # 50 epochs
                eta_min=self.config['training']['scheduler']['eta_min']
            )
        return None

    def _init_metrics(self):
        return {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_recall': [], 'val_recall': [],
            'train_precision': [], 'val_precision': [],
            'train_auroc': [], 'val_auroc': [],
            'train_cmad': [], 'val_cmad': [],  # Track CMAD values
            'learning_rate': []
        }

    def train_epoch(self, epoch):
        """Train for one epoch with frozen backbones"""
        self.model.train()

        train_loss = 0
        train_preds, train_labels = [], []
        train_probs = []
        train_cmad_values = []

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{self.config['training']['epochs']}")

        for frames, labels, _ in progress_bar:
            frames = frames.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Extract features (no gradients for backbones)
                with torch.no_grad():
                    app_feats = self.appearance_model(frames)
                    depth_feats = self.depth_model(frames)
                    motion_feats = self.motion_model(frames)

                # Forward pass through fusion model
                self.optimizer.zero_grad()

                if self.cmad_enabled:
                    outputs, cmad_values, _ = self.model(app_feats, depth_feats, motion_feats, return_cmad=True)
                    train_cmad_values.extend(cmad_values.cpu().numpy().tolist())
                else:
                    outputs = self.model(app_feats, depth_feats, motion_feats)

                loss = self.criterion(outputs, labels)

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_params,
                                               self.config['training']['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_params,
                                               self.config['training']['gradient_clip'])
                self.optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            train_preds.extend(preds.cpu().numpy().tolist())
            train_labels.extend(labels.cpu().numpy().tolist())
            train_probs.extend(probs.cpu().numpy().tolist())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        metrics = {
            'loss': train_loss / len(self.train_loader),
            'predictions': train_preds,
            'labels': train_labels,
            'probabilities': train_probs,
            'cmad': np.mean(train_cmad_values) if train_cmad_values else 0
        }

        return metrics

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()

        val_loss = 0
        val_preds, val_labels = [], []
        val_probs = []
        val_cmad_values = []

        for frames, labels, _ in tqdm(self.val_loader, desc="Validating"):
            frames = frames.to(self.device)
            labels = labels.to(self.device).float()

            app_feats = self.appearance_model(frames)
            depth_feats = self.depth_model(frames)
            motion_feats = self.motion_model(frames)

            if self.cmad_enabled:
                outputs, cmad_values, _ = self.model(app_feats, depth_feats, motion_feats, return_cmad=True)
                val_cmad_values.extend(cmad_values.cpu().numpy().tolist())
            else:
                outputs = self.model(app_feats, depth_feats, motion_feats)

            loss = self.criterion(outputs, labels)

            val_loss += loss.item()
            probs = torch.sigmoid(outputs)
            val_preds.extend((probs > 0.5).float().cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

        metrics = {
            'loss': val_loss / len(self.val_loader),
            'predictions': val_preds,
            'labels': val_labels,
            'probabilities': val_probs,
            'cmad': np.mean(val_cmad_values) if val_cmad_values else 0
        }

        return metrics

    def train(self):
        """Main training loop for 50 epochs with early stopping"""
        print(f"\nStarting training for {self.config['training']['epochs']} epochs...")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print(f"Early stopping patience: {self.config['training']['early_stopping']['patience']}")

        for epoch in range(self.config['training']['epochs']):
            gc.collect()
            torch.cuda.empty_cache()

            # Train
            train_metrics = self.train_epoch(epoch)
            train_stats = self.metrics_calc.calculate(
                train_metrics['predictions'],
                train_metrics['labels'],
                train_metrics['probabilities']
            )

            # Validate
            val_metrics = self.validate()
            val_stats = self.metrics_calc.calculate(
                val_metrics['predictions'],
                val_metrics['labels'],
                val_metrics['probabilities']
            )

            # Update metrics
            self._update_metrics(train_metrics['loss'], val_metrics['loss'],
                                 train_stats, val_stats,
                                 train_metrics['cmad'], val_metrics['cmad'])

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Check for improvement
            current_val_acc = val_stats['accuracy']
            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                self.best_epoch = epoch
                self.no_improve = 0
                save_checkpoint(self.model, self.optimizer, epoch, self.metrics, "best_model.pth")
                logging.info(f"New best model saved with val acc: {self.best_val_acc:.4f}")
            else:
                self.no_improve += 1

            # Logging
            self._log_epoch(epoch, train_metrics['loss'], val_metrics['loss'],
                            train_stats, val_stats,
                            train_metrics['cmad'], val_metrics['cmad'])

            # Early stopping (patience = 10)
            if self.no_improve >= self.config['training']['early_stopping']['patience']:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                save_checkpoint(self.model, self.optimizer, epoch, self.metrics,
                                f"checkpoint_epoch_{epoch + 1}.pth")

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pth"))

        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch + 1}")

        return self.model, self.metrics

    def _update_metrics(self, train_loss, val_loss, train_stats, val_stats, train_cmad, val_cmad):
        """Update metrics dictionary"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_stats['accuracy'])
        self.metrics['val_acc'].append(val_stats['accuracy'])
        self.metrics['train_f1'].append(train_stats['f1'])
        self.metrics['val_f1'].append(val_stats['f1'])
        self.metrics['train_recall'].append(train_stats['recall'])
        self.metrics['val_recall'].append(val_stats['recall'])
        self.metrics['train_precision'].append(train_stats['precision'])
        self.metrics['val_precision'].append(val_stats['precision'])

        if 'auroc' in train_stats:
            self.metrics['train_auroc'].append(train_stats['auroc'])
            self.metrics['val_auroc'].append(val_stats['auroc'])

        self.metrics['train_cmad'].append(train_cmad)
        self.metrics['val_cmad'].append(val_cmad)
        self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

    def _log_epoch(self, epoch, train_loss, val_loss, train_stats, val_stats, train_cmad, val_cmad):
        """Log epoch metrics"""
        logging.info(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logging.info(f"Train Acc: {train_stats['accuracy']:.4f} | Val Acc: {val_stats['accuracy']:.4f}")
        logging.info(f"Train F1: {train_stats['f1']:.4f} | Val F1: {val_stats['f1']:.4f}")
        logging.info(f"Train Recall: {train_stats['recall']:.4f} | Val Recall: {val_stats['recall']:.4f}")
        logging.info(f"Train CMAD: {train_cmad:.6f} | Val CMAD: {val_cmad:.6f}")

        if 'auroc' in train_stats:
            logging.info(f"Train AUROC: {train_stats['auroc']:.4f} | Val AUROC: {val_stats['auroc']:.4f}")

        logging.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")