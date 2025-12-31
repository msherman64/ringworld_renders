import gradio as gr
import numpy as np
import PIL.Image
from .core import Renderer

# Custom CSS to prevent the 'white flash' by hiding the Gradio loading spinner 
# and keeping the old image visible.
CSS = """
.gradio-container { background-color: #0b0f19 !important; color: #e5e7eb !important; }
#output_img { background-color: #0b0f19 !important; border-radius: 8px; overflow: hidden; border: none !important; }
#output_img img { object-fit: contain; }

/* Keep the image fully opaque and sharp while generating */
.generating, .pending { 
    opacity: 1 !important; 
    filter: none !important; 
    transition: none !important;
}

/* Hide ALL Gradio loading indicators, spinners, and progress bars */
.loading, .progress-view, .loader, .spinner { 
    display: none !important; 
    visibility: hidden !important; 
}
"""


def create_ui():

    renderer = Renderer()

    def render_frame(fov, look_x, look_y, look_z, time_hr, 
                     use_atm, use_shad, use_shine, resolution):
        look_at = np.array([look_x, look_y, look_z])
        # Ensure look_at is not zero
        if np.linalg.norm(look_at) < 1e-6:
            look_at = np.array([1.0, 0.0, 0.0])
            
        time_sec = time_hr * 3600.0
        # Maintain 4:3 aspect ratio
        w = resolution
        h = int(resolution * 0.75)
        
        image_data = renderer.render(
            width=w, height=h, fov=fov, look_at=look_at,
            time_sec=time_sec, use_atmosphere=use_atm,
            use_shadows=use_shad,
            use_ring_shine=use_shine
        )
        return PIL.Image.fromarray(image_data)


    with gr.Blocks(title="Ringworld Renderer") as demo:


        gr.Markdown("# Ringworld Renderer: Professional Viewport")
        gr.Markdown("Interactive physical simulation of a Niven-scale Ringworld.")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎥 Camera Settings")
                    fov_slider = gr.Slider(minimum=10, maximum=160, value=95, label="Field of View (FOV)", info="Width of your peripheral vision")
                    look_x = gr.Slider(minimum=-1, maximum=1, value=1.0, step=0.01, label="Look Forward (+Spinward)", info="+X direction")
                    look_y = gr.Slider(minimum=-1, maximum=1, value=1.0, step=0.01, label="Look Up (+Zenith)", info="+Y direction")

                    look_z = gr.Slider(minimum=-1, maximum=1, value=0.0, step=0.01, label="Look Right (+Axial)", info="+Z direction")
                    res_slider = gr.Slider(minimum=128, maximum=1024, value=512, step=128, label="Render Resolution", info="Lower for speed, higher for quality")
                    reset_btn = gr.Button("🔄 Reset Viewport", variant="secondary")

                
                with gr.Group():
                    gr.Markdown("### ☀️ Environment")
                    time_slider = gr.Slider(minimum=0, maximum=24, value=0, step=0.1, label="Time of Day (Hours)", info="Noon is 0h/24h, Midnight is 12h")
                    with gr.Row():
                        atmosphere_toggle = gr.Checkbox(value=True, label="Atmosphere", info="Coupled Scattering & Extinction")
                    with gr.Row():
                        shad_toggle = gr.Checkbox(value=True, label="Shadow Squares", info="Night cycle")
                        shine_toggle = gr.Checkbox(value=True, label="Ring-shine", info="Arch illumination")
            
            with gr.Column(scale=2):
                output_img = gr.Image(label="Physically Accurate Viewport", 
                                    interactive=False, elem_id="output_img")

        
        inputs = [fov_slider, look_x, look_y, look_z, time_slider, 
                  atmosphere_toggle, shad_toggle, shine_toggle, res_slider]

        
        def reset_view():
            return [95, 1.0, 1.0, 0.0, 0.0, True, True, True, 512]



        reset_btn.click(fn=reset_view, outputs=inputs)
        
        # Auto-render on any change
        for input_comp in inputs:
            if hasattr(input_comp, "change"):
                input_comp.change(fn=render_frame, inputs=inputs, outputs=output_img, 
                                 trigger_mode="always_last", show_progress="hidden")

        
        # Initial render
        demo.load(fn=render_frame, inputs=inputs, outputs=output_img, show_progress="hidden")


    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(css=CSS)

