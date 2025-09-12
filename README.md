# Seva Agent: Real-Time Autonomous Prayer Assistant

[![OpenAI Hackathon](https://img.shields.io/badge/OpenAI-Hackathon%202025-blue)](https://devpost.com)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **🏆 OpenAI Open Model Hackathon 2025
> Categories
- For Humanity
- Best Local Agent
- Most Useful Fine-Tune
- Wildcard

Autonomous AI agent that listens to live Sikh prayer services and autonomously displays synchronized Punjabi verses with English meanings, creating immersive spiritual experiences for 30M+ global devotees.

## 🎯 Problem Statement

Younger generations attending Gurdwara (Sikh temple) services understand spoken Punjabi but struggle with:
- Reading Punjabi text in Gurmukhi script
- Understanding authentic spiritual meanings
- Active participation in 2-3 hour prayer services

**Result**: Passive listening without full spiritual engagement or language learning.

## 🚀 Solution

**Seva Agent** transforms prayer experiences by:
- **Real-time ASR**: Listens to live Gurbani recitation
- **Autonomous Display**: Synchronizes projector with original Punjabi text + English meanings
- **Zero Operator**: Eliminates need for manual control during services
- **Educational Impact**: Enhances Punjabi literacy while deepening spiritual connection

## 🏗️ Architecture

```
🎤 Live Audio → 🧠 ASR Engine → 🔍 Ensemble Matching → 🖥️ Desktop Control → 📺 Synchronized Display
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ASR Engine** | Fine-tuned SOTA ASR Models on Religious Texts | Gurmukhi speech recognition |
| **Verse Matching** | Ensemble algorithms | Robust real-time alignment |
| **Desktop Control** | OCR + Socket.IO | Autonomous SikhiToTheMax integration |
| **Navigation** | Anchor/Paath modes | Smart positioning & drift detection |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- macOS (for SikhiToTheMax integration)
- [SikhiToTheMax Desktop App](https://khalisfoundation.org/portfolio/sikhitothemax-everywhere/)
- Microphone access

### Setup

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/sttm-agent.git
cd sttm-agent
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Models**
```bash
python build_index.py  # Builds local verse database
```

4. **Environment Setup**
```bash
cp .env.example .env
# Add your HuggingFace token for model access
echo "HF_TOKEN=your_huggingface_token" >> .env
```

## 🎮 Usage

### Quick Start
```bash
# Run the full autonomous agent
python orchestrator.py --mode agent

# Or run standalone sync mode for testing
python orchestrator.py --mode sync
```

### Manual Control
```bash
# Direct agent execution
python agent_full.py

# Test UI automation
python sttm_ui_controller.py
```

## 📊 Technical Details

### ASR Pipeline
- **Fine-tuning**: 60+ hours curated Gurbani dataset, 10+ epochs
- **Custom Tokenizer and Vocabulary**: Gurmukhi Unicode (U+0A00-U+0A7F)
- **Real-time Processing**: 16kHz, 2-second sliding windows, 1-second overlap

### Ensemble Matching
```python
def ensemble_score(asr_text, ground_truth):
    return weighted_average([
        rapidfuzz.fuzz.partial_ratio(asr_text, ground_truth) * 0.4,
        rapidfuzz.fuzz.token_set_ratio(asr_text, ground_truth) * 0.3,
        difflib.SequenceMatcher(None, asr_text, ground_truth).ratio() * 0.3
    ])
```

### Performance Metrics
- **Latency**: <300ms for ASR on chunk, <100ms for verse identification
- **Accuracy**: 99%+ on domain test set
- **Throughput**: Near Real-time Alignment

## 🎯 Key Features

- ✅ **Autonomous Operation**: Zero human intervention required
- ✅ **Real-time Sync**: Sub-second verse identification and display
- ✅ **Drift Detection**: Automatic recovery from positioning errors  
- ✅ **Leading Prediction**: Anticipates verses for seamless transitions
- ✅ **Cultural Preservation**: Maintains authentic sacred text integrity
- ✅ **Educational Value**: Enhances Punjabi literacy and spiritual engagement

## 🔧 Configuration

### Audio Settings
```python
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
OVERLAP = 1.0
SLIDING_WORDS = 24
```

### Matching Thresholds
```python
CONF_THRESHOLD = 72
PERSISTENCE_REQUIRED = 2
ANCHOR_STRONG_SCORE = 75
LEADING_TRIGGER_SCORE = 55
```

## 📁 Project Structure

```
sttm-agent/
├── agent_full.py              # Main ASR engine
├── orchestrator.py            # System coordinator
├── sttm_ui_controller.py      # Desktop app automation
├── sttm_sync_client.py        # STTM integration wrapper
├── sttm_socketio.py           # Socket.IO communication
├── verse_dataset.py           # Verse-to-shabad mapping
├── build_index.py             # Local database builder
├── fb_mms_1b_fine_tuning.py   # Fine tune ASR model
├── local_banidb/              # Verse database
│   ├── line_store.json        # Verse content
│   └── inverted.json          # Search index
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🎬 Demo

[🎥 **Watch Demo Video**](https://www.youtube.com/watch?v=iLqkGfZPL4c)

### Screenshots
- Live prayer service with synchronized display
- Real-time verse matching in action
- Autonomous desktop control interface

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

## 📈 Impact

- **Global Reach**: Serving 30M+ Sikh devotees worldwide
- **Cultural Preservation**: Digitizing and democratizing sacred text access
- **Educational Value**: Improving Punjabi literacy in younger generations
- **Community Building**: Creating inclusive spiritual experiences
- **Technical Innovation**: Advancing low-resource language ASR

## 🔮 Future Roadmap

- [ ] Mobile app integration
- [ ] Edge optimization for limited compute environments
- [ ] Federated learning across global deployments
- [ ] Multi-language translation (10+ languages)
- [ ] Custom ChatGPTs for personalized religious conversations


## 🏆 Recognition

- **OpenAI Open Model Hackathon 2025** - "For Humanity" Category Submission
- **First-of-its-kind** autonomous Gurbani recognition system
- **Production Deployment** in multiple Gurdwaras worldwide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI**: For the Open Model Hackathon opportunity
- **NVIDIA**: For GPU's for ASR model fine tuning
- **HuggingFace**: For model hosting and datasets platform
- **Khalis Foundation**: For SikhiToTheMax desktop application
- **Sikh Community**: For inspiration and cultural guidance

## 📞 Contact

- **Project Lead**: [Jaspal Singh Saluja](jaspal.saluja@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/jsaluja/sttm-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jsaluja/sttm-agent/discussions)

---

**Built with ❤️ and AI**

*Seva (selfless service) through technology*
