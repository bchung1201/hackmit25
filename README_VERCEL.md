# Mentra Reality Pipeline - Vercel Deployment

A modern web demo of the Mentra Reality Pipeline deployed on Vercel with automatic deployments and global CDN.

## 🚀 Quick Deploy to Vercel

### Option 1: One-Click Deploy
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/bchung1201/hackmit25)

### Option 2: Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Deploy to production
vercel --prod
```

### Option 3: GitHub Integration
1. **Connect your GitHub repository to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect it's a static site

2. **Automatic deployments:**
   - Every push to `main` branch = automatic deployment
   - Preview deployments for pull requests
   - Custom domains supported

## 🎯 Features

### ✅ **Vercel Benefits:**
- **Lightning Fast** - Global CDN with edge caching
- **Automatic HTTPS** - SSL certificates included
- **Custom Domains** - Use your own domain
- **Preview Deployments** - Test before going live
- **Analytics** - Built-in performance monitoring
- **Zero Configuration** - Works out of the box

### ✅ **Demo Features:**
- **Video Upload** - Drag & drop with preview
- **Simulated Processing** - Real-time progress simulation
- **3D Reconstruction Demo** - Shows room and furniture counts
- **Emotion Detection Demo** - Random emotion generation
- **Floor Map Visualization** - Room-by-room analysis
- **Mobile Responsive** - Works on all devices

## 📁 File Structure

```
├── index.html              # Main page
├── styles.css              # Styling
├── script.js               # JavaScript functionality
├── vercel.json             # Vercel configuration
├── package.json            # Project metadata
└── README_VERCEL.md        # This documentation
```

## 🔧 Configuration

### **vercel.json**
- Static site configuration
- Custom routing rules
- Security headers
- Build settings

### **package.json**
- Project metadata
- Scripts for development
- Keywords for discoverability

## 🌐 Deployment URLs

After deployment, you'll get:
- **Production:** `https://mentra-reality-pipeline.vercel.app`
- **Preview:** `https://mentra-reality-pipeline-git-branch.vercel.app`
- **Custom Domain:** `https://yourdomain.com` (if configured)

## 🛠️ Development

### **Local Development:**
```bash
# Install Vercel CLI
npm i -g vercel

# Start development server
vercel dev

# Or use Python server
python -m http.server 8000
```

### **Environment Variables:**
No environment variables needed for the static demo.

## 📊 Performance

- **Core Web Vitals** - Optimized for Google's metrics
- **Edge Caching** - Content served from nearest location
- **Image Optimization** - Automatic image optimization
- **Bundle Analysis** - Built-in bundle size monitoring

## 🔄 Continuous Deployment

1. **Push to GitHub** - Triggers automatic deployment
2. **Preview Deployments** - Every PR gets a preview URL
3. **Production Deployments** - Main branch goes to production
4. **Rollback** - Easy rollback to previous versions

## 🎨 Customization

### **Styling:**
- Edit `styles.css` for visual changes
- Modify `index.html` for content
- Update `script.js` for functionality

### **Vercel Settings:**
- Custom domains in Vercel dashboard
- Environment variables if needed
- Build settings and redirects

## 📱 Mobile Support

- **Responsive Design** - Mobile-first approach
- **Touch Interactions** - Optimized for touch devices
- **Performance** - Fast loading on mobile networks
- **PWA Ready** - Can be converted to Progressive Web App

## 🔍 Analytics

Vercel provides built-in analytics:
- **Page Views** - Track visitor engagement
- **Performance** - Core Web Vitals monitoring
- **Geographic** - See where your users are
- **Real User Monitoring** - Actual performance data

## 🚨 Troubleshooting

### **Common Issues:**

1. **Build Fails:**
   - Check `vercel.json` configuration
   - Ensure all files are committed
   - Check Vercel build logs

2. **404 Errors:**
   - Verify routing in `vercel.json`
   - Check file paths are correct
   - Ensure `index.html` exists

3. **Slow Loading:**
   - Optimize images and assets
   - Check bundle size
   - Use Vercel's performance insights

## 🎉 Success!

Once deployed, your demo will be available at:
`https://mentra-reality-pipeline.vercel.app`

## 📞 Support

- **Vercel Documentation:** [vercel.com/docs](https://vercel.com/docs)
- **Vercel Community:** [github.com/vercel/vercel/discussions](https://github.com/vercel/vercel/discussions)
- **Project Issues:** Create an issue in this repository

---

**Note:** This is a static demo version. For the full-featured version with real backend processing, use the Flask application (`web_app.py`).
