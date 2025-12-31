import gradio as gr
import PIL.Image
import time

from .core import Renderer
from .tools.visualize_shadows import render_system_plot

# Custom CSS
CSS = """
.gradio-container { background-color: #0b0f19 !important; color: #e5e7eb !important; }
#output_img { background-color: #0b0f19 !important; border-radius: 8px; overflow: hidden; border: none !important; }
#output_img img { object-fit: contain; }
#system_img { background-color: #0b0f19 !important; border-radius: 8px; overflow: hidden; border: none !important; }
#system_img img { object-fit: contain; }

.generating, .pending { opacity: 1 !important; filter: none !important; transition: none !important; }
.loading, .progress-view, .loader, .spinner { display: none !important; visibility: hidden !important; }
"""


def create_ui():

    renderer = Renderer()

    def render_frame(fov, yaw, pitch, time_hr, 
                     use_atm, use_shad, use_shine, debug_ss, resolution):
        time_sec = time_hr * 3600.0

        # Maintain 4:3 aspect ratio
        w = resolution
        h = int(resolution * 0.75)
        
        # 1. Main Viewport
        image_data = renderer.render(
            width=w, height=h, fov=fov, yaw=yaw, pitch=pitch,
            time_sec=time_sec, use_atmosphere=use_atm,
            use_shadows=use_shad,
            use_ring_shine=use_shine,
            debug_shadow_squares=debug_ss
        )
        viewport_img = PIL.Image.fromarray(image_data)
        
        # 2. System View (Matplotlib)
        system_img = render_system_plot(time_sec, renderer)
        
        return viewport_img, system_img


    with gr.Blocks(title="Ringworld Renderer") as demo:

        gr.Markdown("# Ringworld Renderer: Professional Viewport")
        gr.Markdown("Interactive physical simulation of a Niven-scale Ringworld.")
        
        with gr.Row():
            # Left Column: Controls (Scale 1)
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎥 Camera Settings")
                    fov_slider = gr.Slider(minimum=10, maximum=160, value=95, label="Field of View (FOV)")
                    yaw_slider = gr.Slider(minimum=-180, maximum=180, value=0, step=1, label="Yaw (Spinward ↔ Axial)", info="0°: Spinward, 90°: Axial Right")
                    pitch_slider = gr.Slider(minimum=-90, maximum=90, value=45, step=1, label="Pitch (Horizon ↔ Zenith)", info="0°: Horizon, 90°: Zenith (Up)")
                    res_slider = gr.Slider(minimum=128, maximum=1024, value=512, step=128, label="Render Resolution")
                    reset_btn = gr.Button("🔄 Reset Viewport", variant="secondary")

                
                with gr.Group():
                    gr.Markdown("### ☀️ Environment")
                    time_slider = gr.Slider(minimum=0, maximum=24, value=0, step=0.1, label="Time of Day (Hours)", info="Noon is 0h/24h, Midnight is 12h")
                    
                    with gr.Row():
                        animate_chk = gr.Checkbox(value=False, label="Animate (24h Cycle)", interactive=True)
                        
                    with gr.Row():
                        atmosphere_toggle = gr.Checkbox(value=True, label="Atmosphere", info="Coupled Scattering")
                    with gr.Row():
                        shad_toggle = gr.Checkbox(value=True, label="Shadow Squares", info="Night cycle")
                        shine_toggle = gr.Checkbox(value=True, label="Ring-shine", info="Arch illumination")
                    with gr.Row():
                        debug_ss_toggle = gr.Checkbox(value=False, label="False Color Shadow Squares", info="Debug Visualization")
            
            # Right Column: Outputs (Scale 2)
            with gr.Column(scale=2):
                output_img = gr.Image(label="Physically Accurate Viewport", interactive=False, elem_id="output_img")
                system_img = gr.Image(label="System View (Top-Down)", interactive=False, elem_id="system_img")

        
        inputs = [fov_slider, yaw_slider, pitch_slider, time_slider, 
                  atmosphere_toggle, shad_toggle, shine_toggle, debug_ss_toggle, res_slider]
        outputs = [output_img, system_img]

        
        def reset_view():
            return [95, 0, 45, 0.0, True, True, True, False, 512]


        reset_btn.click(fn=reset_view, outputs=inputs)
        
        # Auto-render on change
        for input_comp in inputs:
            if hasattr(input_comp, "change"):
                input_comp.change(fn=render_frame, inputs=inputs, outputs=outputs, 
                                 trigger_mode="always_last", show_progress="hidden")
        
        # Animation Loop
        # We use a recursive function or Interval?
        # Gradio events are tricky for infinite loops.
        # We can use `demo.load` with `every` IF we update the component.
        # Or a generator.
        
        def animate(is_animating, current_time):
            if is_animating:
                new_time = (current_time + 0.1) % 24
                return new_time
            return current_time

        # Timer trigger?
        # Use gr.Timer if available in newer gradio, or just chained events.
        # Simple recursion: when timer changes, if animate is True, update time.
        
        # Let's try chained event:
        # Checkbox change -> triggers function?
        # We want to continuously update `time_slider`.
        
        # Using a generator for the output image is standard for animation, but here we want to update the slider too?
        # Simpler: The Checkbox toggles a periodic event.
        
        # Hack: Use a hidden component to debounce/trigger.
        # Actually, let's just make the user manually drag for now IF animation is hard.
        # But User requested "toggle to animate".
        # We'll use a generator on the button click?
        
        def cycle_animation(is_animating, current_time):
            while is_animating:
                current_time = (current_time + 0.05) % 24.0
                yield current_time
                time.sleep(0.05) # ~20fps max

        # When checkbox is True, we run the animation generator
        animate_chk.change(fn=cycle_animation, inputs=[animate_chk, time_slider], outputs=[time_slider])
        

        # Initial render
        demo.load(fn=render_frame, inputs=inputs, outputs=outputs, show_progress="hidden")


    return demo

def main():
    demo = create_ui()
    demo.launch(css=CSS)

if __name__ == "__main__":
    main()

