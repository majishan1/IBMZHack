# IBMZHack Project 2026 - Tokamak Prediction Model
---
## Project Overview

Tokamak disruptions are sudden losses of plasma confinement in fusion reactors, which can lead to significant damage to the reactor components. In order to effectively predict these disruptions, prediction models need to forecast them within a few hundred milliseconds before they form. Our project trains a model using the Tokamark dataset (on the LinuxONE server with PyTorch) and uses it to provide actionable commands to stabilise the Tokamak according to the real-time data stream.