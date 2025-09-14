# Mentra Reality Pipeline - GitHub Pages Demo

A static demo version of the Mentra Reality Pipeline designed for GitHub Pages hosting. This version showcases the frontend interface with simulated backend processing.

## 🎯 Features

### ✅ **Static Demo Interface**
- Beautiful, responsive frontend design
- Video upload and preview functionality
- Simulated real-time processing with progress updates
- Interactive demo results display
- No backend dependencies - works entirely in the browser

### ✅ **GitHub Pages Ready**
- Pure HTML/CSS/JavaScript
- No server-side processing required
- Optimized for static hosting
- Mobile-responsive design

## 🚀 Quick Start

### Option 1: Direct GitHub Pages Deployment
1. **Fork this repository** to your GitHub account
2. **Enable GitHub Pages** in repository settings:
   - Go to Settings → Pages
   - Select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"
3. **Access your site** at `https://yourusername.github.io/hackmit25`

### Option 2: Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/hackmit25.git
cd hackmit25

# Open in browser
open index.html
# or
python -m http.server 8000
# then visit http://localhost:8000
```

## 🌐 Live Demo

Visit the live demo at: `https://yourusername.github.io/hackmit25`

## 📁 File Structure

```
├── index.html              # Main HTML page
├── styles.css              # CSS styles
├── script.js               # JavaScript functionality
├── README_GITHUB_PAGES.md  # This documentation
└── README.md               # Main project documentation
```

## 🎨 Demo Features

### **Video Upload**
- Drag & drop video upload
- Support for MP4, MOV, AVI formats
- Video preview functionality
- File validation

### **Simulated Processing**
- Real-time progress updates
- Simulated emotion detection
- Simulated 3D reconstruction
- Simulated room mapping
- Simulated trajectory tracking

### **Interactive Results**
- 3D Reconstruction statistics
- Emotion analysis results
- Floor map visualization
- Download simulation (demo mode)

## 🔧 Customization

### **Styling**
Edit `styles.css` to customize:
- Colors and themes
- Layout and spacing
- Animations and transitions
- Responsive breakpoints

### **Functionality**
Edit `script.js` to customize:
- Demo simulation parameters
- Processing time and progress
- Result generation logic
- UI interactions

### **Content**
Edit `index.html` to customize:
- Text content and messaging
- Section layout
- Navigation elements
- Call-to-action buttons

## 📱 Mobile Support

The demo is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones
- Various screen sizes

## 🎯 Demo Limitations

Since this is a static demo:
- **No real processing** - All results are simulated
- **No file downloads** - Download buttons show demo messages
- **No backend integration** - No actual 3D reconstruction or emotion detection
- **Client-side only** - All processing happens in the browser

## 🚀 Production Deployment

For a production version with real backend processing:

1. **Use the Flask version** (`web_app.py`)
2. **Deploy to a cloud platform** (Heroku, AWS, Google Cloud)
3. **Set up proper backend infrastructure**
4. **Configure environment variables**
5. **Set up database and file storage**

## 🔍 Browser Compatibility

Tested and works on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📊 Performance

- **Fast loading** - Optimized static files
- **Smooth animations** - CSS3 transitions
- **Responsive design** - Mobile-first approach
- **No external dependencies** - Self-contained

## 🛠️ Development

### **Local Development**
```bash
# Start local server
python -m http.server 8000

# Or use Node.js
npx serve .

# Or use PHP
php -S localhost:8000
```

### **Testing**
- Test video upload functionality
- Verify responsive design
- Check browser compatibility
- Test all interactive elements

## 📝 License

This project is part of the HackMIT 2025 Mentra Reality Pipeline.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Contact the development team
- Check the documentation

---

**Note**: This is a demo version for GitHub Pages. For the full-featured version with real backend processing, use the Flask application (`web_app.py`).
