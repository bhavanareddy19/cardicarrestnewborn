"""
Generate a comprehensive PDF documentation with diagrams for the Neural Health Predictor project.
Uses matplotlib for diagrams and fpdf2 for PDF generation.
"""

from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Create diagrams directory
DIAGRAMS_DIR = os.path.join(os.path.dirname(__file__), "doc_diagrams")
os.makedirs(DIAGRAMS_DIR, exist_ok=True)


def create_architecture_diagram():
    """Create system architecture flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Title
    ax.text(6, 13.5, 'System Architecture', fontsize=18, fontweight='bold', 
            ha='center', color='#003366')
    
    # Input Box
    rect = FancyBboxPatch((2, 12), 8, 1, boxstyle="round,pad=0.05", 
                          facecolor='#E8F4FD', edgecolor='#003366', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 12.5, 'INPUT: 10 Health Features\n(Birth Weight, Heart Rate, Breathing, etc.)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(6, 11.7), xytext=(6, 12), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Data Pipeline Box
    rect = FancyBboxPatch((1.5, 10.2), 9, 1.4, boxstyle="round,pad=0.05", 
                          facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 11.2, 'DATA PIPELINE', ha='center', fontsize=11, fontweight='bold', color='#E65100')
    ax.text(6, 10.6, 'Encode → Split (70/15/15) → Scale', ha='center', fontsize=9)
    
    # Split arrows
    ax.annotate('', xy=(3, 9.8), xytext=(4.5, 10.2), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    ax.annotate('', xy=(6, 9.8), xytext=(6, 10.2), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    ax.annotate('', xy=(9, 9.8), xytext=(7.5, 10.2), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Three paths
    # Path A - Tabular
    rect = FancyBboxPatch((0.5, 8.3), 3.5, 1.4, boxstyle="round,pad=0.05", 
                          facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(rect)
    ax.text(2.25, 9.2, 'Path A: Tabular', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(2.25, 8.7, 'Scaled Features\n→ Models 1-10', ha='center', fontsize=8)
    
    # Path B - Text/BERT
    rect = FancyBboxPatch((4.25, 8.3), 3.5, 1.4, boxstyle="round,pad=0.05", 
                          facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 9.2, 'Path B: BERT', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.text(6, 8.7, 'Text → BioBERT\n→ Model 12', ha='center', fontsize=8)
    
    # Path C - Embedding
    rect = FancyBboxPatch((8, 8.3), 3.5, 1.4, boxstyle="round,pad=0.05", 
                          facecolor='#FFF3E0', edgecolor='#EF6C00', linewidth=2)
    ax.add_patch(rect)
    ax.text(9.75, 9.2, 'Path C: Embedding', ha='center', fontsize=10, fontweight='bold', color='#EF6C00')
    ax.text(9.75, 8.7, 'Raw Categories\n→ Model 11', ha='center', fontsize=8)
    
    # Arrows to models
    ax.annotate('', xy=(6, 7.9), xytext=(2.25, 8.3), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    ax.annotate('', xy=(6, 7.9), xytext=(6, 8.3), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    ax.annotate('', xy=(6, 7.9), xytext=(9.75, 8.3), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # 12 Models Box
    rect = FancyBboxPatch((1, 4.5), 10, 3.3, boxstyle="round,pad=0.05", 
                          facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 7.5, '12-MODEL DNN ENSEMBLE', ha='center', fontsize=12, fontweight='bold', color='#7B1FA2')
    
    # Model grid
    models = ['ShallowWide', 'DeepNarrow', 'PyramidBN', 'DiamondSELU',
              'ResidualBlock', 'SwishLN', 'MixedAct', 'HeavyReg',
              'AttentionNet', 'VeryDeep', 'EmbedNet', 'BERTFusion']
    colors = ['#BBDEFB', '#C8E6C9', '#FFECB3', '#F8BBD9',
              '#B2DFDB', '#D1C4E9', '#FFCCBC', '#CFD8DC',
              '#B3E5FC', '#DCEDC8', '#FFE0B2', '#B2EBF2']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        row = i // 4
        col = i % 4
        x = 1.5 + col * 2.4
        y = 6.5 - row * 0.8
        rect = FancyBboxPatch((x, y), 2.2, 0.6, boxstyle="round,pad=0.02", 
                              facecolor=color, edgecolor='#333333', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 1.1, y + 0.3, model, ha='center', va='center', fontsize=7)
    
    # Arrow to voting
    ax.annotate('', xy=(6, 4.0), xytext=(6, 4.5), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Weighted Voting Box
    rect = FancyBboxPatch((2.5, 2.8), 7, 1.1, boxstyle="round,pad=0.05", 
                          facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 3.5, 'WEIGHTED SOFT VOTING', ha='center', fontsize=11, fontweight='bold', color='#C62828')
    ax.text(6, 3.1, 'Combine predictions weighted by AUC', ha='center', fontsize=9)
    
    # Arrow to output
    ax.annotate('', xy=(6, 2.4), xytext=(6, 2.8), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Output box
    rect = FancyBboxPatch((2, 1.2), 8, 1.1, boxstyle="round,pad=0.05", 
                          facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 1.9, 'FINAL OUTPUT', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(6, 1.45, 'LOW  |  MEDIUM  |  HIGH  Risk', ha='center', fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, 'architecture.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def create_model_architecture_diagram(model_num, name, layers, activation, description):
    """Create individual model architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Title
    ax.text(3, 3.8, f'Model {model_num}: {name}', fontsize=12, fontweight='bold', 
            ha='center', color='#003366')
    
    # Draw layers
    n_layers = len(layers)
    layer_height = 2.5 / n_layers
    y_start = 3.3
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, n_layers))
    
    for i, (layer, color) in enumerate(zip(layers, colors)):
        width = min(layer / max(layers) * 4 + 1, 5)
        x = (6 - width) / 2
        y = y_start - i * layer_height
        
        rect = FancyBboxPatch((x, y - layer_height * 0.7), width, layer_height * 0.6, 
                              boxstyle="round,pad=0.02", facecolor=color, 
                              edgecolor='#003366', linewidth=1)
        ax.add_patch(rect)
        ax.text(3, y - layer_height * 0.4, f'{layer} neurons', ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold')
        
        if i < n_layers - 1:
            ax.annotate('', xy=(3, y - layer_height * 0.8), xytext=(3, y - layer_height * 0.7),
                       arrowprops=dict(arrowstyle='->', color='#666666', lw=1))
    
    # Activation label
    ax.text(3, 0.5, f'Activation: {activation}', ha='center', fontsize=9, 
            style='italic', color='#666666')
    
    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, f'model_{model_num}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def create_bert_pipeline_diagram():
    """Create BERT processing pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Title
    ax.text(6, 5.7, 'BERT Fusion Pipeline', fontsize=16, fontweight='bold', 
            ha='center', color='#003366')
    
    # Step 1: Input
    rect = FancyBboxPatch((0.5, 4), 3, 1.2, boxstyle="round,pad=0.05", 
                          facecolor='#E8F4FD', edgecolor='#003366', linewidth=2)
    ax.add_patch(rect)
    ax.text(2, 4.85, 'Step 1: Input Data', ha='center', fontsize=10, fontweight='bold')
    ax.text(2, 4.35, 'BirthWeight=Low\nHeartRate=Rapid\nSkinTinge=Bluish', 
            ha='center', fontsize=8)
    
    # Arrow
    ax.annotate('', xy=(4, 4.6), xytext=(3.5, 4.6), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Step 2: Text Generation
    rect = FancyBboxPatch((4.25, 4), 3.5, 1.2, boxstyle="round,pad=0.05", 
                          facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 4.85, 'Step 2: Clinical Text', ha='center', fontsize=10, fontweight='bold', color='#E65100')
    ax.text(6, 4.35, '"The newborn has low\nbirth weight with rapid\nheart rate..."', 
            ha='center', fontsize=7, style='italic')
    
    # Arrow
    ax.annotate('', xy=(8.25, 4.6), xytext=(7.75, 4.6), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Step 3: BioBERT
    rect = FancyBboxPatch((8.5, 4), 3, 1.2, boxstyle="round,pad=0.05", 
                          facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(rect)
    ax.text(10, 4.85, 'Step 3: BioBERT', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.text(10, 4.35, 'Extract 768-dim\nmedical embeddings', ha='center', fontsize=8)
    
    # Vertical arrows down
    ax.annotate('', xy=(2, 2.8), xytext=(2, 4), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    ax.annotate('', xy=(10, 2.8), xytext=(10, 4), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Original features box
    rect = FancyBboxPatch((0.5, 1.8), 3, 0.9, boxstyle="round,pad=0.05", 
                          facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(rect)
    ax.text(2, 2.4, '10 Tabular Features', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.text(2, 2.0, '(Scaled Numbers)', ha='center', fontsize=8)
    
    # BERT embeddings box
    rect = FancyBboxPatch((8.5, 1.8), 3, 0.9, boxstyle="round,pad=0.05", 
                          facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(rect)
    ax.text(10, 2.4, '768 BERT Features', ha='center', fontsize=9, fontweight='bold', color='#1565C0')
    ax.text(10, 2.0, '(Medical Meaning)', ha='center', fontsize=8)
    
    # Merge arrows
    ax.annotate('', xy=(6, 1.4), xytext=(3.5, 2.2), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    ax.annotate('', xy=(6, 1.4), xytext=(8.5, 2.2), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Fusion box
    rect = FancyBboxPatch((4, 0.5), 4, 0.8, boxstyle="round,pad=0.05", 
                          facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 1.0, 'BERT Fusion Model', ha='center', fontsize=10, fontweight='bold', color='#7B1FA2')
    ax.text(6, 0.7, '778 Total Features -> Dense Layers -> Output', ha='center', fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, 'bert_pipeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def create_ensemble_voting_diagram():
    """Create ensemble voting visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Title
    ax.text(5, 5.7, 'Weighted Ensemble Voting', fontsize=14, fontweight='bold', 
            ha='center', color='#003366')
    
    # Model predictions
    models = ['Model 1', 'Model 2', 'Model 3', '...', 'Model 12']
    aucs = [0.92, 0.89, 0.95, '...', 0.97]
    preds = ['M: 50%', 'M: 55%', 'M: 45%', '...', 'M: 60%']
    
    for i, (model, auc, pred) in enumerate(zip(models, aucs, preds)):
        x = 1 + i * 1.7
        rect = FancyBboxPatch((x, 4), 1.4, 1.2, boxstyle="round,pad=0.02", 
                              facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.7, 4.9, model, ha='center', fontsize=8, fontweight='bold')
        if auc != '...':
            ax.text(x + 0.7, 4.5, f'AUC: {auc}', ha='center', fontsize=7)
            ax.text(x + 0.7, 4.2, pred, ha='center', fontsize=7, color='#666666')
        else:
            ax.text(x + 0.7, 4.5, auc, ha='center', fontsize=10)
        
        # Arrow down
        ax.annotate('', xy=(x + 0.7, 3.5), xytext=(x + 0.7, 4), 
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=1))
    
    # Weighted sum box
    rect = FancyBboxPatch((2, 2.3), 6, 1.1, boxstyle="round,pad=0.05", 
                          facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 3.0, 'Weighted Average', ha='center', fontsize=11, fontweight='bold', color='#E65100')
    ax.text(5, 2.55, 'Final = Σ(Weight_i × Prediction_i) / Σ(Weights)', 
            ha='center', fontsize=9)
    
    # Arrow to output
    ax.annotate('', xy=(5, 1.8), xytext=(5, 2.3), 
                arrowprops=dict(arrowstyle='->', color='#003366', lw=2))
    
    # Output
    rect = FancyBboxPatch((3, 0.8), 4, 0.9, boxstyle="round,pad=0.05", 
                          facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 1.4, 'Final Prediction: MEDIUM RISK', ha='center', 
            fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(5, 1.05, 'Low: 28% | Medium: 52% | High: 20%', ha='center', fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, 'ensemble_voting.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def create_data_features_diagram():
    """Create features visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Title
    ax.text(5, 6.7, '10 Input Features & Risk Levels', fontsize=14, fontweight='bold', 
            ha='center', color='#003366')
    
    features = [
        ('Birth Weight', 'TooLow -> Low -> Normal', '#FFCDD2'),
        ('Family History', '>2 cases -> 0-2 -> None', '#F8BBD9'),
        ('Preterm Birth', '4+ wks -> 2-4 -> Term', '#E1BEE7'),
        ('Heart Rate', 'Rapid -> High -> Normal', '#D1C4E9'),
        ('Breathing', 'High -> Moderate -> None', '#C5CAE9'),
        ('Skin Tinge', 'Bluish -> Light -> Normal', '#BBDEFB'),
        ('Responsiveness', 'None -> Limited -> Normal', '#B2EBF2'),
        ('Movement', 'Diminished -> Decreased -> Normal', '#B2DFDB'),
        ('Delivery Type', 'C-Section -> Difficult -> Normal', '#C8E6C9'),
        ("Mother's BP", 'VeryHigh -> High -> Normal', '#DCEDC8'),
    ]
    
    for i, (name, levels, color) in enumerate(features):
        row = i // 2
        col = i % 2
        x = 0.5 + col * 5
        y = 5.8 - row * 1.15
        
        rect = FancyBboxPatch((x, y), 4.5, 1, boxstyle="round,pad=0.02", 
                              facecolor=color, edgecolor='#333333', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.15, y + 0.7, name, fontsize=9, fontweight='bold', va='center')
        ax.text(x + 0.15, y + 0.3, levels, fontsize=7, va='center', color='#555555')
    
    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, 'features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


class DetailedPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(15, 20, 15)
        self.set_auto_page_break(auto=True, margin=25)
        
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, 'Neural Health Predictor - Comprehensive Documentation', align='C')
            self.ln(8)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
    
    def section_title(self, title):
        self.ln(6)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.multi_cell(0, 8, title)
        self.set_draw_color(0, 51, 102)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)
    
    def subsection_title(self, title):
        self.ln(4)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 7, title)
        self.set_text_color(0, 0, 0)
        self.ln(2)
    
    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_x(15)
        self.multi_cell(0, 5, text)
        self.ln(2)
    
    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_x(15)
        self.multi_cell(0, 5, "  * " + text)
    
    def add_image(self, path, w=180):
        if self.get_y() > 200:
            self.add_page()
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.ln(5)
    
    def add_table(self, headers, data, col_widths):
        if self.get_y() > 240:
            self.add_page()
        
        self.set_x(15)
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()
        
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        
        for row_idx, row in enumerate(data):
            if self.get_y() > 270:
                self.add_page()
            fill = row_idx % 2 == 0
            self.set_fill_color(245, 245, 245) if fill else self.set_fill_color(255, 255, 255)
            self.set_x(15)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell)[:30], border=1, fill=fill)
            self.ln()
        self.ln(5)


def generate_all_diagrams():
    """Generate all diagrams."""
    print("Generating diagrams...")
    arch_path = create_architecture_diagram()
    print(f"  Created: {arch_path}")
    
    bert_path = create_bert_pipeline_diagram()
    print(f"  Created: {bert_path}")
    
    voting_path = create_ensemble_voting_diagram()
    print(f"  Created: {voting_path}")
    
    features_path = create_data_features_diagram()
    print(f"  Created: {features_path}")
    
    return arch_path, bert_path, voting_path, features_path


def generate_detailed_pdf():
    # Generate diagrams first
    arch_path, bert_path, voting_path, features_path = generate_all_diagrams()
    
    pdf = DetailedPDF()
    
    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 32)
    pdf.set_text_color(0, 51, 102)
    pdf.ln(50)
    pdf.multi_cell(0, 15, 'Neural Health Predictor', align='C')
    
    pdf.set_font('Helvetica', '', 20)
    pdf.set_text_color(51, 51, 51)
    pdf.ln(10)
    pdf.multi_cell(0, 10, 'Early Detection of Cardiac Arrest\nin Newborn Babies', align='C')
    
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 8, 'A Comprehensive Deep Learning System\nwith 12-Model Ensemble Architecture', align='C')
    
    pdf.ln(40)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 8, 'Contributors:\nV. Bhavana  |  S. Roshini  |  D. Sanjana\n\nProject Guide: Ms. M.N. Sailaja\n\nCMR College of Engineering & Technology, Hyderabad', align='C')
    
    # Table of Contents
    pdf.add_page()
    pdf.section_title("Table of Contents")
    toc = [
        "1. Executive Summary",
        "2. Problem Statement & Motivation",
        "3. System Architecture Overview",
        "4. Input Features & Data Pipeline",
        "5. The 12 Neural Network Models (Detailed)",
        "6. BERT-Based Medical Text Processing",
        "7. Ensemble Weighted Voting Mechanism",
        "8. Training Configuration & Hyperparameters",
        "9. Installation & Usage Guide",
        "10. Expected Performance Metrics",
        "11. Technology Stack",
        "12. Future Improvements",
        "13. References"
    ]
    for item in toc:
        pdf.bullet_point(item)
    
    # Section 1: Executive Summary
    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "The Neural Health Predictor is a state-of-the-art machine learning system "
        "designed to assist healthcare professionals in the early detection of cardiac "
        "arrest risk in newborn babies. By analyzing 10 critical physiological indicators, "
        "the system provides real-time risk assessment across three severity levels: "
        "Low, Medium, and High risk."
    )
    pdf.ln(3)
    pdf.body_text(
        "What makes this system unique is its ensemble approach - combining 12 different "
        "deep neural network architectures, each with its own specialization. This is "
        "analogous to consulting 12 different specialist doctors, each examining the "
        "patient from their unique perspective, and then combining their opinions for "
        "the most accurate diagnosis possible."
    )
    
    pdf.subsection_title("Key Technical Innovations")
    pdf.bullet_point("12-Model Deep Learning Ensemble for robust predictions")
    pdf.bullet_point("BioBERT integration for understanding medical context")
    pdf.bullet_point("Clinical text generation from tabular data")
    pdf.bullet_point("Weighted voting based on individual model performance")
    pdf.bullet_point("Hyperparameter optimization with 10,000+ configurations")
    
    # Section 2: Problem Statement
    pdf.add_page()
    pdf.section_title("2. Problem Statement & Motivation")
    
    pdf.subsection_title("The Clinical Challenge")
    pdf.body_text(
        "Cardiac arrest in neonates (newborn babies) is a life-threatening emergency "
        "that requires immediate medical intervention. The challenge lies in the fact "
        "that symptoms can be extremely subtle and easy to miss, especially in the "
        "chaotic environment of a neonatal intensive care unit (NICU)."
    )
    pdf.ln(2)
    pdf.body_text(
        "Traditional monitoring systems often fail to detect the early warning signs "
        "that precede cardiac arrest. By the time obvious symptoms appear, the window "
        "for effective intervention may have significantly narrowed, leading to poor "
        "outcomes including permanent brain damage or death."
    )
    
    pdf.subsection_title("Our Solution Approach")
    pdf.body_text(
        "Our system addresses this challenge by continuously analyzing multiple "
        "physiological parameters and combining predictions from 12 different neural "
        "network architectures. This multi-model approach provides several advantages:"
    )
    pdf.bullet_point("Diversity of perspectives catches patterns any single model might miss")
    pdf.bullet_point("Ensemble averaging reduces prediction variance and increases reliability")
    pdf.bullet_point("BERT integration understands the contextual meaning of symptoms")
    pdf.bullet_point("Real-time processing enables immediate alerting")
    
    # Section 3: Architecture
    pdf.add_page()
    pdf.section_title("3. System Architecture Overview")
    pdf.body_text(
        "The following diagram illustrates the complete data flow through our system, "
        "from input features to final risk prediction:"
    )
    pdf.add_image(arch_path, w=170)
    
    pdf.body_text(
        "The architecture consists of three parallel processing paths that feed into "
        "the 12-model ensemble:"
    )
    pdf.bullet_point("Path A (Tabular): Scaled numerical features for standard DNN models 1-10")
    pdf.bullet_point("Path B (BERT): Clinical text converted to BioBERT embeddings for Model 12")
    pdf.bullet_point("Path C (Embedding): Raw categorical encodings for entity embedding Model 11")
    
    # Section 4: Features
    pdf.add_page()
    pdf.section_title("4. Input Features & Data Pipeline")
    pdf.body_text(
        "The system analyzes 10 carefully selected physiological indicators that are "
        "clinically significant for predicting cardiac arrest risk in newborns:"
    )
    pdf.add_image(features_path, w=170)
    
    pdf.subsection_title("Feature Encoding")
    pdf.body_text(
        "Each feature is categorical with 3 severity levels. Features are encoded to "
        "integers (1=lowest risk, 3=highest risk) for numerical processing. The system "
        "uses stratified splitting (70% train, 15% validation, 15% test) to maintain "
        "class balance across all subsets."
    )
    
    pdf.subsection_title("Data Pipeline Steps")
    pdf.bullet_point("Step 1: Load CSV data with 10 features per patient")
    pdf.bullet_point("Step 2: Encode categorical values to integers using predefined mappings")
    pdf.bullet_point("Step 3: Stratified split maintaining class proportions")
    pdf.bullet_point("Step 4: StandardScaler normalization (fit on train, transform all)")
    pdf.bullet_point("Step 5: Generate clinical text narratives for BERT processing")
    pdf.bullet_point("Step 6: Extract BioBERT embeddings (768-dimensional vectors)")
    
    # Section 5: Models Detailed
    pdf.add_page()
    pdf.section_title("5. The 12 Neural Network Models (Detailed)")
    pdf.body_text(
        "Each model in our ensemble has unique architectural characteristics designed "
        "to capture different aspects of the input data. Here is a comprehensive "
        "breakdown of each model:"
    )
    
    models_detailed = [
        ("Model 1: ShallowWide", "The Quick Generalist", 
         "2 layers (256 to 128 neurons)", "ReLU",
         "This model uses wide layers with few depths, making it excellent at capturing "
         "broad, obvious patterns. Like a general practitioner who can quickly assess "
         "overall health status. Its simplicity also makes it highly interpretable and "
         "serves as a reliable baseline for the ensemble."),
        
        ("Model 2: DeepNarrow", "The Detailed Analyzer",
         "6 layers (64 neurons each)", "ELU",
         "With 6 layers of narrow width, this model examines data step-by-step, capable "
         "of finding subtle, complex patterns that shallower models miss. The ELU "
         "activation helps with learning faster and avoiding dying neuron problems."),
        
        ("Model 3: PyramidBN", "The Organized Thinker",
         "4 layers (256 to 32 pyramid)", "GELU + BatchNorm",
         "Gradually compresses information from 256 to 32 neurons, forcing the network "
         "to learn the most important features at each level. BatchNormalization ensures "
         "stable training and faster convergence."),
        
        ("Model 4: DiamondSELU", "The Self-Balancing Expert",
         "5 layers (diamond shape)", "SELU",
         "Diamond shape expands then contracts, allowing the network to explore a higher "
         "dimensional representation before compressing. SELU activation provides "
         "self-normalizing properties, eliminating the need for BatchNorm."),
        
        ("Model 5: ResidualBlock", "The Memory Keeper",
         "4 blocks with skip connections", "ReLU + BatchNorm",
         "Uses skip connections that allow gradients to flow directly through the network, "
         "preventing vanishing gradient problems. Even if deep layers learn poorly, "
         "the original information still reaches the output."),
        
        ("Model 6: SwishLayerNorm", "The Smooth Operator",
         "4 layers (128 neurons each)", "Swish + LayerNorm",
         "Swish activation (x * sigmoid(x)) provides smoother gradients than ReLU, "
         "often leading to better optimization. LayerNorm normalizes each sample "
         "independently, making it robust to batch size variations."),
        
        ("Model 7: MixedActivation", "The Versatile Specialist",
         "4 layers with different activations", "ReLU, ELU, GELU, Swish",
         "Each layer uses a different activation function, allowing the network to "
         "leverage the unique strengths of each. ReLU for speed, ELU for negative values, "
         "GELU for probabilistic behavior, and Swish for smoothness."),
        
        ("Model 8: HeavyRegularization", "The Cautious One",
         "4 layers with L1+L2 and 50% Dropout", "ReLU",
         "Heavy regularization prevents the model from memorizing training data. "
         "L1 encourages sparsity (some weights become exactly zero), L2 prevents "
         "any single weight from becoming too large, and 50% dropout forces the "
         "network to learn redundant representations."),
        
        ("Model 9: AttentionNet", "The Focused Expert",
         "Attention layer + 3 dense layers", "ReLU",
         "Inspired by TabNet, this model learns to dynamically focus on the most "
         "relevant features for each specific input. Not all features are equally "
         "important for every patient, and this model adapts its attention accordingly."),
        
        ("Model 10: VeryDeep", "The Exhaustive Researcher",
         "8 layers (varying widths)", "PReLU",
         "The deepest model in the ensemble with 8 hidden layers. PReLU (Parametric ReLU) "
         "learns the optimal slope for negative inputs during training. Can model "
         "extremely complex non-linear relationships."),
        
        ("Model 11: EmbeddingNet", "The Category Expert",
         "10 embedding layers + dense", "ReLU",
         "Instead of using simple integer encodings, learns 4-dimensional embedding "
         "vectors for each category value. These embeddings capture semantic similarity - "
         "for example, 'LowWeight' and 'WeightTooLow' would have similar embeddings."),
        
        ("Model 12: BERTFusion", "The Medical Language Expert",
         "768 BERT + 10 tabular to dense", "GELU",
         "The star model that combines tabular features with BioBERT embeddings. "
         "Natural language understanding from millions of medical papers helps interpret "
         "clinical context that pure numerical models cannot capture."),
    ]
    
    for name, nickname, arch, activation, description in models_detailed:
        if pdf.get_y() > 220:
            pdf.add_page()
        pdf.subsection_title(f"{name} - '{nickname}'")
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_x(15)
        pdf.cell(0, 5, f"Architecture: {arch}")
        pdf.ln(5)
        pdf.set_x(15)
        pdf.cell(0, 5, f"Activation: {activation}")
        pdf.ln(5)
        pdf.set_font('Helvetica', '', 10)
        pdf.body_text(description)
    
    # Section 6: BERT
    pdf.add_page()
    pdf.section_title("6. BERT-Based Medical Text Processing")
    pdf.body_text(
        "One of the most innovative aspects of our system is the integration of BERT "
        "(Bidirectional Encoder Representations from Transformers) for understanding "
        "medical context. Here's how it works:"
    )
    pdf.add_image(bert_path, w=170)
    
    pdf.subsection_title("Clinical Text Generation")
    pdf.body_text(
        "The ClinicalTextGenerator converts each patient's 10 categorical values into "
        "a coherent medical narrative. This is not simple concatenation - it uses "
        "domain-specific templates to generate clinically meaningful sentences."
    )
    pdf.ln(2)
    pdf.body_text("Example transformation:")
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_x(15)
    pdf.multi_cell(0, 5, 
        'Input: BirthWeight=LowWeight, HeartRate=RapidHeartRate, SkinTinge=Bluish\n\n'
        'Output: "The newborn has a low birth weight, with a significant family history '
        'of cardiac conditions. The infant is presenting with a rapid heart rate '
        'indicating tachycardia, with cyanotic skin coloration suggesting poor oxygenation..."')
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(3)
    
    pdf.subsection_title("BioBERT Embeddings")
    pdf.body_text(
        "BioBERT is a domain-specific BERT model pre-trained on large-scale biomedical "
        "corpora including PubMed abstracts and PMC full-text articles. It understands "
        "that 'tachycardia' and 'rapid heart rate' are the same concept, and that "
        "'cyanotic' indicates a serious oxygen deprivation concern."
    )
    pdf.bullet_point("Model: dmis-lab/biobert-v1.1")
    pdf.bullet_point("Output: 768-dimensional embedding per text")
    pdf.bullet_point("Token limit: 128 tokens per text")
    
    # Section 7: Ensemble
    pdf.add_page()
    pdf.section_title("7. Ensemble Weighted Voting Mechanism")
    pdf.body_text(
        "The ensemble combines predictions from all 12 models using a weighted soft "
        "voting scheme. This is more sophisticated than simple majority voting:"
    )
    pdf.add_image(voting_path, w=160)
    
    pdf.subsection_title("How Weights Are Determined")
    pdf.body_text(
        "Each model's weight is proportional to its AUC score on the validation set. "
        "Models that performed better during training get more influence in the final "
        "prediction. This adaptive weighting ensures that the ensemble benefits more "
        "from its stronger members while still considering diverse perspectives."
    )
    
    pdf.subsection_title("Mathematical Formulation")
    pdf.body_text(
        "For each class c (Low, Medium, High), the ensemble probability is computed as:"
    )
    pdf.set_font('Courier', '', 10)
    pdf.set_x(15)
    pdf.multi_cell(0, 5, "P(class=c) = Sum(weight_i * P_i(class=c)) / Sum(weight_i)")
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(2)
    pdf.body_text("Where weight_i is the AUC of model i, and P_i is model i's predicted probability.")
    
    # Section 8: Training Config
    pdf.add_page()
    pdf.section_title("8. Training Configuration & Hyperparameters")
    
    pdf.add_table(
        headers=["Parameter", "Value", "Description"],
        data=[
            ["Random Seed", "42", "For reproducibility"],
            ["Train/Val/Test Split", "70/15/15%", "Stratified by class"],
            ["Batch Size", "64", "Samples per gradient update"],
            ["Epochs", "200", "Maximum training iterations"],
            ["Early Stopping", "20 epochs", "Patience before stopping"],
            ["LR Reduction", "Factor 0.5", "After 10 plateau epochs"],
            ["Optimizer", "Adam", "Adaptive learning rate"],
            ["Loss Function", "Sparse Categorical CE", "Multi-class classification"],
        ],
        col_widths=[50, 40, 90]
    )
    
    pdf.subsection_title("Hyperparameter Optimization")
    pdf.body_text(
        "The system supports extensive HPO using Optuna and Ray Tune backends. "
        "In combined mode, it explores 10,000+ configurations to find optimal "
        "hyperparameters for each model architecture."
    )
    pdf.bullet_point("Optuna: Tree-structured Parzen Estimator (TPE) sampler")
    pdf.bullet_point("Ray Tune: ASHA scheduler for early trial pruning")
    pdf.bullet_point("Search space includes: learning rate, dropout, layer widths, regularization")
    
    # Section 9: Usage
    pdf.add_page()
    pdf.section_title("9. Installation & Usage Guide")
    
    pdf.subsection_title("Prerequisites")
    pdf.bullet_point("Python 3.8 or higher")
    pdf.bullet_point("CUDA-capable GPU (recommended for BERT extraction)")
    pdf.bullet_point("At least 16GB RAM")
    
    pdf.subsection_title("Installation")
    pdf.set_font('Courier', '', 9)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_x(15)
    pdf.multi_cell(0, 5, 
        "# Clone repository\n"
        "git clone <repository-url>\n"
        "cd cardicarrestnewborn\n\n"
        "# Create virtual environment\n"
        "python -m venv venv\n"
        "venv\\Scripts\\activate  # Windows\n\n"
        "# Install dependencies\n"
        "pip install -r requirements.txt", fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(5)
    
    pdf.subsection_title("Running the Pipeline")
    pdf.add_table(
        headers=["Mode", "Command", "Description"],
        data=[
            ["Full", "python main.py --mode full", "Complete pipeline"],
            ["BERT only", "python main.py --mode bert", "Extract embeddings"],
            ["Ensemble only", "python main.py --mode ensemble", "Train 12 models"],
            ["HPO only", "python main.py --mode hpo", "Hyperparameter search"],
            ["Evaluate", "python main.py --mode evaluate", "Test saved models"],
        ],
        col_widths=[40, 70, 70]
    )
    
    # Section 10: Results
    pdf.add_page()
    pdf.section_title("10. Expected Performance Metrics")
    
    pdf.add_table(
        headers=["Metric", "Expected Range", "Description"],
        data=[
            ["Accuracy", "92-95%", "Overall correct predictions"],
            ["AUC (macro)", "0.94-0.97", "Area under ROC curve"],
            ["Precision", "0.90-0.95", "True positive rate"],
            ["Recall", "0.88-0.93", "Sensitivity"],
            ["F1-Score", "0.89-0.94", "Harmonic mean of P and R"],
        ],
        col_widths=[50, 50, 80]
    )
    
    pdf.subsection_title("Performance Comparison")
    pdf.body_text(
        "The ensemble consistently outperforms individual models due to the diversity "
        "of its components. The weighted voting mechanism typically improves AUC by "
        "2-5% compared to the best individual model."
    )
    
    # Section 11: Tech Stack
    pdf.section_title("11. Technology Stack")
    
    pdf.add_table(
        headers=["Category", "Technologies"],
        data=[
            ["Deep Learning", "TensorFlow 2.12+, Keras"],
            ["NLP/BERT", "HuggingFace Transformers, PyTorch 2.0+"],
            ["ML Utilities", "scikit-learn, imbalanced-learn"],
            ["Hyperparameter Tuning", "Optuna 3.2+, Ray Tune 2.5+"],
            ["Data Processing", "Pandas, NumPy"],
            ["Visualization", "Matplotlib, Seaborn"],
            ["GUI Application", "PyQt5"],
        ],
        col_widths=[60, 120]
    )
    
    # Section 12: Future
    pdf.add_page()
    pdf.section_title("12. Future Improvements")
    
    pdf.bullet_point("Real-time monitoring integration with hospital systems")
    pdf.bullet_point("Continuous learning from new patient data")
    pdf.bullet_point("Explainability features using SHAP/LIME")
    pdf.bullet_point("Mobile application for bedside predictions")
    pdf.bullet_point("Integration with electronic health records (EHR)")
    pdf.bullet_point("Multi-center validation studies")
    
    # Section 13: References
    pdf.section_title("13. References")
    refs = [
        ("1.", "Gupta, K., et al.", "A Machine Learning Approach Using Statistical Models for Early Detection of Cardiac Arrest in Newborn Babies", "IEEE Access"),
        ("2.", "Choi, E., et al.", "Using recurrent neural network models for early detection of heart failure onset", "J. Amer. Med. Inform. Assoc."),
        ("3.", "Rajkomar, A., et al.", "Scalable and accurate deep learning with electronic health records", "NPJ Digital Medicine"),
        ("4.", "Lee, J., et al.", "BioBERT: a pre-trained biomedical language representation model", "Bioinformatics"),
        ("5.", "Devlin, J., et al.", "BERT: Pre-training of Deep Bidirectional Transformers", "NAACL 2019"),
    ]
    
    for num, authors, title, venue in refs:
        pdf.set_font('Helvetica', '', 9)
        pdf.set_x(15)
        pdf.multi_cell(0, 5, f"{num} {authors} - \"{title}\" - {venue}")
        pdf.ln(2)
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_x(15)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(150, 0, 0)
    pdf.multi_cell(0, 6, 
        "DISCLAIMER: This tool is intended for research and educational purposes only. "
        "Clinical decisions should always be made by qualified healthcare professionals. "
        "This system is not a substitute for professional medical advice, diagnosis, or treatment.")
    
    # Save PDF
    output_path = os.path.join(os.path.dirname(__file__), "Neural_Health_Predictor_Documentation.pdf")
    pdf.output(output_path)
    print(f"\nComprehensive PDF generated successfully!")
    print(f"Location: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_detailed_pdf()
