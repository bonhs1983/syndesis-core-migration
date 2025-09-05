HRV_FEATURES = ['meanRR','SDNN','RMSSD','LF','HF','LF_HF']
CHAPTERS = ['Breathing','Stress','Recovery','Energy','Focus','Balance']

def mapping_function():
    """Return a toy mapping chapter -> HRV feature"""
    return dict(zip(CHAPTERS, HRV_FEATURES))
