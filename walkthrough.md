# Neural Health Predictor: Cardiac Arrest Detection in Newborns

## Complete Project Documentation

---

# 📌 What Does This Project Do?

**Simple Explanation:** This system acts like a smart medical assistant that looks at a newborn baby's health information and predicts if they might be at risk for cardiac arrest.

**How it works:**
1. **Takes in baby's health data** → 10 different health measurements
2. **Processes it through 12 different AI brains** → Each "brain" looks at the data differently
3. **Combines all opinions** → Like getting a second (or 12th!) opinion from doctors
4. **Gives a risk level** → Low, Medium, or High risk

---

# 🏗️ System Architecture (How Everything Connects)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATA                                      │
│                         (Baby's Health Info)                                 │
│   Birth Weight, Heart Rate, Breathing, Skin Color, Responsiveness, etc.    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PROCESSING                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ Convert Text    │    │ Split Data      │    │ Normalize Numbers       │  │
│  │ to Numbers      │───▶│ Train/Val/Test  │───▶│ (Scale 0-1 range)       │  │
│  │ (Encoding)      │    │ (70/15/15%)     │    │                         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│     PATH A: NUMBERS ONLY      │   │     PATH B: TEXT + NUMBERS            │
│                               │   │                                       │
│   10 health measurements      │   │  Convert numbers to medical story:    │
│   go directly to Models 1-10  │   │  "Baby has low birth weight,          │
│                               │   │   rapid heart rate, bluish skin..."   │
│                               │   │                                       │
│                               │   │         ▼                             │
│                               │   │   Run through BioBERT AI to           │
│                               │   │   extract medical meaning             │
│                               │   │         ▼                             │
│                               │   │   Goes to Model 12 (BERT Fusion)      │
└───────────────────────────────┘   └───────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        12 AI MODELS (THE ENSEMBLE)                          │
│                                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Model 1  │ │Model 2  │ │Model 3  │ │Model 4  │ │Model 5  │ │Model 6  │   │
│  │Shallow  │ │Deep     │ │Pyramid  │ │Diamond  │ │Residual │ │Swish    │   │
│  │Wide     │ │Narrow   │ │BN       │ │SELU     │ │Block    │ │LayerNorm│   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │           │         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Model 7  │ │Model 8  │ │Model 9  │ │Model 10 │ │Model 11 │ │Model 12 │   │
│  │Mixed    │ │Heavy    │ │Attention│ │Very     │ │Embedding│ │BERT     │   │
│  │Activation││Regulariz│ │Net      │ │Deep     │ │Net      │ │Fusion   │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │           │         │
│       └───────────┴───────────┴─────┬─────┴───────────┴───────────┘         │
│                                     │                                       │
│                                     ▼                                       │
│                         ┌───────────────────────┐                           │
│                         │   WEIGHTED VOTING     │                           │
│                         │   Combine all 12      │                           │
│                         │   predictions         │                           │
│                         └───────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FINAL OUTPUT                                      │
│                                                                             │
│              ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│              │    LOW     │  │   MEDIUM   │  │    HIGH    │                 │
│              │    RISK    │  │    RISK    │  │    RISK    │                 │
│              │    ✓       │  │     ⚠      │  │     ⚠⚠     │                 │
│              └────────────┘  └────────────┘  └────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 📊 What Data Does the System Use?

The system analyzes **10 health factors** about newborn babies:

| # | Feature | What It Measures | Danger Levels (High → Low Risk) |
|---|---------|-----------------|--------------------------------|
| 1 | **Birth Weight** | How heavy the baby is at birth | WeightTooLow → LowWeight → NormalWeight |
| 2 | **Family History** | Heart problems in the family | AboveTwoCases → ZeroToTwoCases → NoCases |
| 3 | **Preterm Birth** | How early the baby was born | 4+ weeks early → 2-4 weeks → Full term |
| 4 | **Heart Rate** | How fast the heart is beating | Rapid → High → Normal |
| 5 | **Breathing Difficulty** | Trouble breathing | High difficulty → Some → None |
| 6 | **Skin Tinge** | Skin color (bluish = low oxygen) | Bluish → Light Bluish → Normal |
| 7 | **Responsiveness** | Reacts to touch/sound | Unresponsive → Semi → Responsive |
| 8 | **Movement** | How much the baby moves | Diminished → Decreased → Normal |
| 9 | **Delivery Type** | How baby was born | C-Section → Difficult → Normal |
| 10 | **Mother's BP History** | Mom's blood pressure | Very High → High → Normal |

---

# 🧠 The 12 AI Models Explained (In Simple Terms)

## Why 12 Different Models?

Think of it like asking 12 different doctors for their opinion. Each doctor has a different specialty and way of thinking. By combining all their opinions, we get a more reliable diagnosis than asking just one.

---

## Model 1: ShallowWide 
**The Quick Generalist**

```
INPUT (10 features)
        │
        ▼
┌───────────────────┐
│   256 neurons     │  ← Wide layer (many neurons)
│   ReLU activation │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   128 neurons     │
│   ReLU activation │
└─────────┬─────────┘
          │
          ▼
    OUTPUT (3 classes)
```

**What it does:** Uses only 2 layers but each layer is WIDE (many neurons). Like a doctor who looks at the big picture quickly without overthinking.

**Why it's useful:** Fast, efficient, and surprisingly accurate for simple patterns. Acts as a reliable baseline.

---

## Model 2: DeepNarrow
**The Detailed Analyzer**

```
INPUT (10 features)
        │
        ▼
┌─────────────────┐
│   64 neurons    │
│   ELU activation│
└────────┬────────┘
         │  (6 layers total)
         ▼
┌─────────────────┐
│   64 neurons    │
│   ELU activation│
└────────┬────────┘
         │
    ... (4 more layers)
         │
         ▼
    OUTPUT (3 classes)
```

**What it does:** Uses 6 layers but each is NARROW (fewer neurons). Like a doctor who examines step-by-step, looking for subtle details.

**Why it's useful:** Can find complex, hidden patterns that simpler models miss.

---

## Model 3: PyramidBN
**The Organized Thinker**

```
INPUT (10 features)
        │
        ▼
┌───────────────────┐
│   256 neurons     │  ← Widest
│   BatchNorm + GELU│
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   128 neurons     │  ← Narrower
│   BatchNorm + GELU│
└─────────┬─────────┘
          ▼
┌───────────────────┐
│    64 neurons     │  ← Even narrower
│   BatchNorm + GELU│
└─────────┬─────────┘
          ▼
┌───────────────────┐
│    32 neurons     │  ← Narrowest (pyramid tip)
│   BatchNorm + GELU│
└─────────┬─────────┘
          ▼
    OUTPUT (3 classes)
```

**What it does:** Starts wide and gets narrower like a pyramid. Uses BatchNormalization to keep training stable.

**Why it's useful:** Gradually compresses information, keeping only the most important patterns. Very stable training.

---

## Model 4: DiamondSELU
**The Self-Balancing Expert**

```
INPUT (10 features)
        │
        ▼
┌───────────────────┐
│    64 neurons     │  ← Narrow start
│   SELU activation │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   128 neurons     │  ← Expanding
│   SELU activation │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   256 neurons     │  ← WIDEST (diamond middle)
│   SELU activation │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   128 neurons     │  ← Contracting
│   SELU activation │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│    64 neurons     │  ← Narrow end
│   SELU activation │
└─────────┬─────────┘
          ▼
    OUTPUT (3 classes)
```

**What it does:** Expands then contracts like a diamond shape. SELU activation automatically normalizes the network.

**Why it's useful:** Self-normalizing means it trains well without extra tricks. Good at finding both simple and complex patterns.

---

## Model 5: ResidualBlock
**The Memory Keeper**

```
INPUT (10 features)
        │
        ├─────────────────────────────────┐
        ▼                                 │
┌───────────────────┐                     │ (Skip Connection)
│   128 neurons     │                     │
│   ReLU + BN       │                     │
└─────────┬─────────┘                     │
          │                               │
          ▼                               │
┌───────────────────┐                     │
│   128 neurons     │                     │
│   ReLU + BN       │                     │
└─────────┬─────────┘                     │
          │                               │
          ▼                               │
     ┌────┴────┐                          │
     │  ADD ←──┼──────────────────────────┘
     └────┬────┘
          │
          ▼
    OUTPUT (3 classes)
```

**What it does:** Has "skip connections" that let information bypass layers. Original input is ADDED back to the processed output.

**Why it's useful:** Prevents the "vanishing gradient" problem. Even if deep layers learn poorly, the original info still gets through.

---

## Model 6: SwishLayerNorm
**The Smooth Operator**

```
INPUT (10 features)
        │
        ▼
┌───────────────────┐
│   128 neurons     │
│   Swish activation│ ← Smooth, curved activation
│   LayerNorm       │ ← Normalizes each sample
└─────────┬─────────┘
          │
    (3 more layers)
          │
          ▼
    OUTPUT (3 classes)
```

**What it does:** Uses Swish activation (x × sigmoid(x)) which is smoother than ReLU. LayerNorm normalizes each sample individually.

**Why it's useful:** Smooth activation = smoother learning curves. Often works better for complex patterns.

---

## Model 7: MixedActivation
**The Versatile Specialist**

```
INPUT (10 features)
        │
        ▼
┌───────────────────┐
│   Layer 1: ReLU   │  ← Simple, fast
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   Layer 2: ELU    │  ← Handles negatives better
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   Layer 3: GELU   │  ← Probabilistic approach
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   Layer 4: Swish  │  ← Smooth, modern
└─────────┬─────────┘
          ▼
    OUTPUT (3 classes)
```

**What it does:** Uses DIFFERENT activation functions at each layer instead of the same one everywhere.

**Why it's useful:** Each activation function is good at different things. Mixing them captures more diverse patterns.

---

## Model 8: HeavyRegularization
**The Cautious One**

```
INPUT (10 features)
        │
        ▼
┌───────────────────────────────┐
│   Layer 1                     │
│   L1 + L2 Regularization      │ ← Penalizes large weights
│   Dropout 50%                 │ ← Randomly ignores neurons
└─────────────┬─────────────────┘
              │
         (more layers with same approach)
              │
              ▼
    OUTPUT (3 classes)
```

**What it does:** Heavily penalizes the model for "memorizing" the training data. Forces it to learn general patterns.

**Why it's useful:** Prevents overfitting. Even if other models memorize training data, this one stays generalized.

---

## Model 9: AttentionNet
**The Focused Expert**

```
INPUT (10 features)
        │
        ▼
┌─────────────────────────────────────┐
│         ATTENTION LAYER             │
│                                     │
│   "Which features matter most       │
│    for THIS specific patient?"      │
│                                     │
│   Birth Weight:    ████████ 80%     │
│   Heart Rate:      ██████── 60%     │
│   Skin Tinge:      ████──── 40%     │
│   (learns weights automatically)    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│     Process weighted features       │
└─────────────────┬───────────────────┘
                  │
                  ▼
    OUTPUT (3 classes)
```

**What it does:** Inspired by TabNet. Learns to "pay attention" to the most important features for each patient.

**Why it's useful:** Not all features matter equally for every patient. This model adapts its focus dynamically.

---

## Model 10: VeryDeep
**The Exhaustive Researcher**

```
INPUT (10 features)
        │
        ▼
┌─────────────────┐
│   Layer 1       │
│   PReLU         │
└────────┬────────┘
         │
┌─────────────────┐
│   Layer 2       │
│   PReLU         │
└────────┬────────┘
         │
    ... (8 layers total)
         │
┌─────────────────┐
│   Layer 8       │
│   PReLU         │
└────────┬────────┘
         │
         ▼
    OUTPUT (3 classes)
```

**What it does:** Uses 8 hidden layers (the deepest in our ensemble) with PReLU activation (learnable slope for negatives).

**Why it's useful:** Can model extremely complex relationships. When combined with other models, adds depth to the ensemble.

---

## Model 11: EmbeddingNet
**The Category Expert**

```
INPUT: Raw categories (not scaled numbers!)

   Birth Weight:    "LowWeight"    →    ┌─────┐
                                        │ 4D  │ ← Learned representation
                                        │embed│
                                        └──┬──┘
   Heart Rate:      "RapidHeartRate" →  ┌─────┐
                                        │ 4D  │
                                        │embed│
                                        └──┬──┘
   ... (10 total features)               │
                                        │
                    Combine all embeddings
                                        │
                                        ▼
                                   Dense layers
                                        │
                                        ▼
                               OUTPUT (3 classes)
```

**What it does:** Instead of using simple numbers (1, 2, 3), learns a rich 4-dimensional "embedding" for each category value.

**Why it's useful:** Categories like "LowWeight" and "WeightTooLow" are related - embeddings can capture this similarity.

---

## Model 12: BERTFusion (The Star Model)
**The Medical Language Expert**

```
STEP 1: Convert numbers to medical text
┌──────────────────────────────────────────────────────────────────────────┐
│ INPUT DATA:                                                               │
│ BirthWeight=LowWeight, HeartRate=Rapid, SkinTinge=Bluish, ...           │
│                                                                          │
│                              ▼                                           │
│                                                                          │
│ OUTPUT TEXT:                                                             │
│ "The newborn has a low birth weight, with a significant family          │
│  history of cardiac conditions. The infant is presenting with a         │
│  rapid heart rate indicating tachycardia, exhibiting severe             │
│  respiratory distress, with cyanotic skin coloration suggesting         │
│  poor oxygenation..."                                                   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 2: Run through BioBERT (trained on medical research)
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   Medical Text  ───▶  [BioBERT]  ───▶  768 numbers representing         │
│                                        the medical meaning               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 3: Combine with original tabular data
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ┌────────────────┐     ┌────────────────┐                             │
│   │ 10 tabular     │     │ 768 BERT       │                             │
│   │ features       │  +  │ embeddings     │  = 778 total features       │
│   └──────┬─────────┘     └──────┬─────────┘                             │
│          │                      │                                        │
│          └──────────┬───────────┘                                        │
│                     ▼                                                    │
│              Dense layers                                                │
│                     │                                                    │
│                     ▼                                                    │
│              OUTPUT (3 classes)                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**What it does:** 
1. Converts the patient's data into a medical story
2. Uses BioBERT (trained on millions of medical papers) to understand the medical meaning
3. Combines this understanding with the raw numbers

**Why it's useful:** BioBERT understands medical language. It knows that "tachycardia" and "rapid heart rate" mean the same thing, and that "cyanotic" indicates serious oxygen problems.

---

# 🎯 How Are All 12 Predictions Combined?

```
┌────────────────────────────────────────────────────────────────────────┐
│                     WEIGHTED SOFT VOTING                                │
│                                                                        │
│  Each model gives probabilities:                                       │
│                                                                        │
│  Model 1:   Low: 30%   Medium: 50%   High: 20%    Weight: 0.92        │
│  Model 2:   Low: 25%   Medium: 55%   High: 20%    Weight: 0.89        │
│  Model 3:   Low: 35%   Medium: 45%   High: 20%    Weight: 0.95        │
│  ...                                                                   │
│  Model 12:  Low: 20%   Medium: 60%   High: 20%    Weight: 0.97        │
│                                                                        │
│  Final = Weighted Average of all model predictions                     │
│                                                                        │
│  FINAL:     Low: 28%   Medium: 52%   High: 20%   → MEDIUM RISK        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

Weight = How well that model performed on validation data (measured by AUC)
```

---

# 🔧 How the BERT Text Generation Works

The `ClinicalTextGenerator` converts each patient's 10 categorical values into 4 sentences:

| Sentence | Features Included | Example Output |
|----------|-------------------|----------------|
| 1 | Birth Weight + Family History + Preterm | "The newborn has a low birth weight, with family history of cardiac conditions, born 2-4 weeks premature." |
| 2 | Heart Rate + Breathing + Skin | "The infant is presenting with elevated heart rate, moderate breathing difficulty, with mild cyanotic skin." |
| 3 | Responsiveness + Movement | "The infant shows limited responsiveness with decreased motor activity." |
| 4 | Delivery Type + Mother's BP | "Delivery was difficult. The mother has history of high blood pressure." |

---

# 🚀 Quick Start Guide

```bash
# 1. Install everything
pip install -r requirements.txt

# 2. Run the full pipeline (takes a while - extracts BERT, trains 12 models, runs HPO)
python main.py --mode full

# 3. Or run individual steps:
python main.py --mode bert      # Just extract BERT embeddings
python main.py --mode ensemble  # Just train the 12 models
python main.py --mode evaluate  # Just evaluate saved models
```

---

# 📈 What Results to Expect

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 92-95% |
| AUC (macro) | 0.94-0.97 |
| Precision | 0.90-0.95 |
| Recall | 0.88-0.93 |

---

# 👥 Team

| Name | Role |
|------|------|
| V. Bhavana | Developer |
| S. Roshini | Developer |
| D. Sanjana | Developer |

**Guide**: Ms. M.N. Sailaja, CMR College of Engineering & Technology

---

> ⚠️ **Disclaimer**: This is a research tool for educational purposes. Always consult medical professionals for actual medical decisions.
