gfx read elements "Brainstem.vtk.exf"

gfx modify g_element "/" general clear;
gfx modify g_element "/" points domain_nodes coordinate data_coordinates tessellation default_points LOCAL glyph sphere size "0.2*0.2*0.2" offset 0,0,0 font default select_on material gold selected_material default_selected render_shaded;

gfx create window
gfx edit scene