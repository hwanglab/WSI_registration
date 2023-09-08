# Registration of Whole Slide Images(WSIs)
![Copy of Final_Manuscript_fig_1](https://github.com/hwanglab/WSI_registration/assets/52568892/5f5091bc-c762-4f42-aa00-c510e0529f60)

## 1. Registering H&E and IHC Slides
  ### How to run
  you can answer target & source image path by following questions;
  - "1 for Dir or 0 for image path:" < you can run this code with multiple images (1) or single image (0) >
  - Enter a ref. image path: < target image path (e.g. H&E path), it should be a directory path containing reference images when you choose 1 above >
  - Enter a moving image path: < source image path (e.g. IHC path), it should be a directory path containing source images when you choose 1 above >
  - Enter an output path: < registered image path, it should be a directory path containing output images when you choose 1 above >
  ### Inside code
  - prefix : output directory
  - fixed_path : target image path (e.g. H&E path)
  - moving_path : source image path (e.g. IHC path)
    
  
## 2. Coordinating Between Source and Target Slides

- **Instructions**: [Check out the usage guide here](https://github.com/hwanglab/WSI_registration/blob/main/map_coords/README.md).
  
- **Demonstration**: [See this example](https://github.com/hwanglab/WSI_registration/blob/main/mapping_coordinate_example.ipynb).

- **Visualizing Mapped Tiles**: To display the mapped tiles on a new canvas for reverse engineering, [refer to this example](https://github.com/hwanglab/WSI_registration/blob/main/draw_mapped_images_example.ipynb).

## License
© This project is licensed under Apache License  Version 2.0 - see the [LICENSE](https://github.com/hwanglab/WSI_registration/blob/main/LICENSE) file for details.

## Contact
Reach out to the [Hwang Lab](https://www.hwanglab.org/).

<div align="center">
    <img src="https://github.com/hwanglab/WSI_registration/assets/52568892/48957936-acb0-4445-b7e9-9dd74e05d0ac" alt="hwanglab_mayo" width="600"/>
</div>
