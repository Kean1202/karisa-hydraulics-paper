# Karisa's Chemical Engineering ML Project 💕

> *A comprehensive machine learning analysis of distillation column performance*
> *Made with love for the smartest engineer in the world*

---

## 📚 What This Project Does

This project analyzes distillation column tray performance using machine learning to:

1. **Identify which variables cause hydraulic failures** (weeping & flooding)
2. **Find dangerous operating conditions to avoid** (risk assessment)
3. **Determine which variables optimize product quality** (conversion & purity)
4. **Discover the best operating conditions** (optimal region analysis)

Think of it as a comprehensive guide to running the perfect distillation column! ✨

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Your amazing brain (already installed ✓)

### Installation

1. **Clone or download this project**
   ```bash
   cd path/to/Karisa
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **You're ready to go!** 🎉

---

## 🎯 How to Run

### Phase 1: Exploratory Data Analysis
```bash
python eda.py
```
Generates beautiful visualizations in the `eda_plots/` folder.

### Phase 2: Hydraulic Analysis
```bash
python hydraulics_a1_a2.py  # Variable importance for WEEP & FLOOD
python hydraulics_b1_b2.py  # Risk assessment for combinations
```
Results saved to: `results/hydraulics/Hydraulics_Analysis_Results.xlsx`

### Phase 3: Quality Analysis
```bash
python quality_a3_a4.py    # Variable importance for CONV & PURITY
python quality_b3_b4.py    # Optimal combination search
```
Results saved to: `results/quality/Quality_Analysis_Results.xlsx`

### Secret Surprise 💝
```bash
python super_important_code.py
```
(Trust me on this one 😊)

---

## 📁 Project Structure

```
Karisa/
├── data/
│   ├── AmAc_Tray.xlsx             # Your experimental data
│   └── karisa_paper.xlsx          # (Original filename)
├── eda.py                         # Exploratory data analysis
├── utils.py                       # Shared utility functions
├── hydraulics_a1_a2.py           # Phase 2: Variable importance (hydraulics)
├── hydraulics_b1_b2.py           # Phase 2: Risk assessment
├── quality_a3_a4.py              # Phase 3: Variable importance (quality)
├── quality_b3_b4.py              # Phase 3: Optimal combinations
├── super_important_code.py       # ❤️
├── MODELING_PLAN.md              # Detailed project roadmap
├── requirements.txt              # Python dependencies
└── README.md                     # You are here!
```

---

## 🔬 The Analysis Pipeline

### Phase 1: Understanding the Data ✅
- Visualizations and statistical tests
- Correlation analysis
- Distribution plots for all variables

### Phase 2: Hydraulic Behavior Analysis ✅
- **Goal A1**: Rank variables that cause WEEP
- **Goal A2**: Rank variables that cause FLOOD
- **Goal B1**: Identify high-risk combinations for WEEP
- **Goal B2**: Identify high-risk combinations for FLOOD

### Phase 3: Quality Optimization 🎯
- **Goal A3**: Rank variables that influence CONVERSION
- **Goal A4**: Rank variables that influence PURITY
- **Goal B3**: Find combinations with highest CONVERSION
- **Goal B4**: Find combinations with highest PURITY
- **Combined**: Find the optimal region (high in BOTH)

### Phase 4: Final Reporting 📊
- Master summary document
- Model comparison tables
- Final visualizations

---

## 📊 Key Features

- ✨ **5-fold cross-validation** for robust model evaluation
- 🎯 **Multiple model comparison** (Linear, Ridge, PLS, Random Forest, XGBoost)
- 📈 **Proper importance ranking** using rank-based averaging
- 🎨 **Beautiful visualizations** for presentations
- 📑 **Clean Excel outputs** with organized sheets
- 🔍 **Risk zone categorization** (percentile-based)
- 💎 **Optimal region detection** for quality metrics

---

## 🤓 Technical Details

### Independent Variables (7)
- `NHOLES` - Number of holes
- `HDIAM` - Hole diameter
- `TRAYSPC` - Tray spacing
- `WEIRHT` - Weir height
- `DECK` - Deck area
- `DIAM` - Column diameter
- `NPASS` - Number of passes

### Dependent Variables
- **Hydraulic**: DESC (PASS/WEEP/FLOOD)
- **Quality**: CONV (Conversion), PURITY (Purity)

### Models Used

**Classification (Hydraulics):**
- Logistic Regression
- Ridge Classifier
- SVM
- Random Forest Classifier
- XGBoost Classifier

**Regression (Quality):**
- Linear Regression
- Ridge Regression
- PLS Regression
- Random Forest Regressor
- XGBoost Regressor

---

## 💡 Tips for Success

1. **Always run scripts in order**: EDA → Hydraulics → Quality
2. **Check the Excel files**: All results are neatly organized
3. **Read MODELING_PLAN.md**: Detailed methodology and progress tracking
4. **Look at the plots**: Visual insights are powerful!
5. **Stay hydrated**: You're doing amazing work! 💧

---

## 🎓 For Your Report/Presentation

Key outputs to include:
- Variable importance rankings from all 4 goals (A1, A2, A3, A4)
- Model comparison tables (show which model performed best)
- Risk zone distributions (how many combinations are high-risk?)
- Optimal region visualization (CONV vs PURITY scatter plot)
- Top operating conditions for maximizing quality

---

## 📝 Notes

- All filtering uses a **whitelist approach** (only valid experimental values kept)
- Cross-validation ensures **robust and reliable** results
- RMSE/MAE statistics are **properly computed per-fold** (not approximations)
- Risk categories use **percentiles** (top 10%, 70-90%, etc.)
- Combined scores use **simple averaging** (avoids compression issues)

---

## 🆘 Troubleshooting

**Issue**: Missing dependencies
**Fix**: `pip install -r requirements.txt`

**Issue**: File not found
**Fix**: Make sure you're running scripts from the project root directory

**Issue**: Plots not displaying
**Fix**: Check if matplotlib backend is configured correctly

**Issue**: Feeling overwhelmed
**Fix**: Take a break, you've got this! ☕

---

## 🌟 You've Got This!

This project represents:
- ✅ Advanced machine learning techniques
- ✅ Rigorous statistical analysis
- ✅ Clean, reproducible code
- ✅ Professional documentation
- ✅ Thoughtful experimental design

You're not just running code - you're doing real chemical engineering research with cutting-edge ML tools. Be proud of this work!

Remember: Even the best engineers take it one step at a time. You're already amazing, and this project is going to be incredible. 💪

---

## 📬 Questions?

- Check `MODELING_PLAN.md` for detailed methodology
- Review the code comments (they're very friendly!)
- Look at the plots (pictures > words sometimes!)

---

**Made with ❤️ for Karisa**
*Keep being brilliant, you incredible human being!*

---

## License

This project is for educational and research purposes.
The data belongs to Karisa's research.
The love and encouragement? Unlimited and open-source. 💕
