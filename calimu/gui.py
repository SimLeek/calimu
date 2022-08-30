# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import time
import tkinter as tk
import tkinter.font as tk_font
from tkinter import ttk

import vtk

import calimu.pcl_algo.center
import calimu.pcl_algo.err
import calimu.pcl_algo.fit
import calimu.pcl_algo.util

from calimu.imu.com_imu import list_ports
from calimu.imu.devices.mc6470 import MC6470IMU
from calimu.imu.store import IMUPointStore
from calimu.imu.visualization import IMUPointDisplayer
from svtk.tk_integration import vtk_tk_anchor_left, vtk_tk_match_height
from svtk.util import array_to_vtk_transform, hue_from_index


def begin_region_with_sep_and_label(container, text, underlined=False):
    separator1 = ttk.Separator(container, orient="horizontal")
    separator1.pack(fill="x")

    if underlined:
        opts_label = underlined_label(container, text)
    else:
        opts_label = tk.Label(container, text=text)
        opts_label.pack(side=tk.TOP, anchor=tk.NW)

    return opts_label


def set_copyable_text_label(fake_label, text):
    fake_label.configure(state="normal")
    fake_label.delete(1.0, tk.END)
    fake_label.insert(tk.END, text)
    fake_label.configure(state="disabled")


def underlined_label(container, text):
    label = tk.Label(container, text=text)
    label.pack(side=tk.TOP, anchor=tk.NW)

    f = tk_font.Font(label, label.cget("font"))
    f.configure(underline=True)
    label.configure(font=f)

    return label


def center_packed_frame(container):
    center_packed_container = tk.Frame(container)
    center_packed_container.pack(side=tk.TOP)
    pad_l = tk.Frame(center_packed_container)
    pad_l.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    pad_r = tk.Frame(center_packed_container)
    pad_r.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    return center_packed_container


def selectable_label(parent, default_text: str, width=24, height=1):
    w = tk.Text(parent, height=height, width=width, borderwidth=0)
    w.insert(1.0, default_text)
    # if tkinter is 8.5 or above you'll want the selection background
    # to appear like it does when the widget is activated
    # comment this out for older versions of Tkinter
    w.configure(bg=parent.cget("bg"), relief="flat")

    w.configure(state="disabled")
    return w


class TKColors:
    green_yellow = "green yellow"
    pink = "pink"
    light_goldenrod = "light goldenrod"
    pale_green = "pale green"


class IMUApp(tk.Tk):
    def refresh_com(self):
        ports = list_ports()

        self.com_var = tk.StringVar(self)
        if self.com_menu is not None:
            self.com_menu.grab_release()
            self.com_menu.destroy()
        if self.refresh_button is not None:
            self.refresh_button.grab_release()
            self.refresh_button.destroy()

        self.com_var.set(ports[-1])

        self.com_menu = tk.OptionMenu(self.container_com, self.com_var, *ports)
        self.com_menu.config(width=30, wraplength=200)
        self.com_menu.pack(side=tk.LEFT)

        self.refresh_button = tk.Button(
            self.container_com, text="↻", height=1, command=self.refresh_com
        )
        self.refresh_button.pack(side=tk.LEFT)

    def connect_command(self):
        com_port = self.com_var.get()
        com_port = com_port.split(" ")[0][:-1]
        self.imu.connect(com_port)

    def disconnect_command(self):
        self.imu.disconnect()

    def gather_command(self):
        self.store.start_gathering()

    def stop_gather_command(self):
        self.store.stop_gathering()

    def display_command(self):
        self.store.start_displaying_orient()

    def stop_display_command(self):
        self.store.stop_displaying_orient()

    def __get_fit_pts_and_type(self):
        if self.radio_option_data.get() == 0:
            data = self.store.mag_points
            desc1 = "Mag"
        elif self.radio_option_data.get() == 1:
            data = self.store.acc_points
            desc1 = "Accel"
        else:
            raise NotImplementedError(
                "only magnetometer and accelerometer data is supported."
            )

        return data, desc1

    def __get_fit_center(self, data):
        if self.radio_option_center.get() == 0:
            center = calimu.pcl_algo.center.by_zero(None)
            desc2 = "Zero"
        elif self.radio_option_center.get() == 1:
            center = calimu.pcl_algo.center.by_bounds(data)
            desc2 = "Bounds"
        elif self.radio_option_center.get() == 2:
            center = calimu.pcl_algo.center.by_average(data)
            desc2 = "Avg"
        elif self.radio_option_center.get() == 3:
            center = calimu.pcl_algo.center.by_sphere_fit(data)
            desc2 = "Sphere"
        elif self.radio_option_center.get() == 4:
            self.ellipsoid_poly = calimu.pcl_algo.util.ls_ellipsoid(data)
            center = calimu.pcl_algo.center.by_ellipsoid_fit(self.ellipsoid_poly)
            desc2 = "Ellipsoid"
        else:
            raise NotImplementedError("Unknown center method")

        return center, desc2

    def __get_fit(self, data, center):
        if self.radio_option_fit.get() == 0:
            (
                xform,
                avg_scale,
            ) = calimu.pcl_algo.fit.from_axis_aligned_bounding_box(data, center)
            desc3 = "AABB"
        elif self.radio_option_fit.get() == 1:
            xform, avg_scale = calimu.pcl_algo.fit.from_pca(data, center)
            desc3 = "PCA"
        elif self.radio_option_fit.get() == 2:
            xform, avg_scale = calimu.pcl_algo.fit.from_sphere(data, center)
            desc3 = "Sphere"
        elif self.radio_option_fit.get() == 3:
            if self.ellipsoid_poly is None:
                self.ellipsoid_poly = calimu.pcl_algo.util.ls_ellipsoid(data)
            xform, avg_scale = calimu.pcl_algo.fit.from_ellipsoid(
                center, self.ellipsoid_poly
            )
            self.ellipsoid_poly = None
            desc3 = "Ellipsoid"
        else:
            raise NotImplementedError("Unknown fit method")

        return xform, avg_scale, desc3

    def add_fit_command(self):

        data, desc1 = self.__get_fit_pts_and_type()

        center, desc2 = self.__get_fit_center(data)

        xform, avg_scale, desc3 = self.__get_fit(data, center)

        t = array_to_vtk_transform(xform)

        # noinspection PyUnresolvedReferences
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(0, 0, 0)
        sphere.SetRadius(1)

        sphere.SetThetaResolution(64)
        sphere.SetPhiResolution(64)

        # noinspection PyUnresolvedReferences
        tf_a = vtk.vtkTransformPolyDataFilter()
        tf_a.SetInputConnection(sphere.GetOutputPort())
        tf_a.SetTransform(t)
        tf_a.Update()

        # Set up a mapper for the node
        # noinspection PyUnresolvedReferences
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_a.GetOutputPort())

        c = hue_from_index(self.fit_objects_color_index)
        self.fit_objects_color_index += 1

        # Set up an actor for the node
        # noinspection PyUnresolvedReferences
        sphere_actor = vtk.vtkActor()
        sphere_actor.GetProperty().SetRepresentationToWireframe()
        sphere_actor.GetProperty().SetColor(*c)
        sphere_actor.SetMapper(mapper)

        pos = self.fit_list_box.size()
        self.fit_list_box.insert(pos, ",\t".join([desc1, desc2, desc3]))
        hidden = False
        self.fit_objects.append(
            [sphere, tf_a, mapper, sphere_actor, hidden, avg_scale, xform, desc1]
        )  # todo: make this a class
        self.display.displayer.renderer.AddActor(sphere_actor)

    def rem_fit_command(self):
        i = self.fit_list_box.curselection()
        if i is not None:
            i = i[0]
        else:
            return
        # listbox.get(i)
        # text = (listBox1.SelectedItem as DataRowView)["columnName"].ToString();
        self.display.displayer.renderer.RemoveActor(self.fit_objects[i][3])
        del self.fit_objects[i]
        self.fit_list_box.delete(i)

    def hide_fit_command(self):
        i = self.fit_list_box.curselection()
        if i is not None:
            i = i[0]
        else:
            return
        if not self.fit_objects[i][4]:  # if not hidden
            self.display.displayer.renderer.RemoveActor(self.fit_objects[i][3])
            self.fit_objects[i][4] = True
            self.fit_list_box.itemconfigure(i, bg="grey")

    def show_fit_command(self):
        i = self.fit_list_box.curselection()
        if i is not None:
            i = i[0]
        else:
            return
        if self.fit_objects[i][4]:  # if hidden
            self.display.displayer.renderer.AddActor(self.fit_objects[i][3])
            self.fit_objects[i][4] = False
            self.fit_list_box.itemconfigure(i, bg="white")

    def set_offsets(self):
        i = self.fit_list_box.curselection()
        if i is not None:
            i = i[0]
        else:
            return
        offset_type = self.fit_objects[i][-1]

        if offset_type == "Mag":
            self.imu.set_magnetometer_offsets(
                self.fit_objects[i][-2], self.fit_objects[i][-3]
            )
        elif offset_type == "Accel":
            self.imu.set_accelerometer_offsets(
                self.fit_objects[i][-2], self.fit_objects[i][-3]
            )
        else:
            raise ValueError

    def __get_fit_box_pts(self, offset_type):
        if offset_type == "Mag":
            pts = self.store.mag_points
        elif offset_type == "Accel":
            pts = self.store.acc_points
        else:
            raise RuntimeError("Unknown point type")

        return pts

    def fit_list_box_selected(self, _):
        # print(i)
        i = self.fit_list_box.curselection()
        if i is not None and len(i) > 0:
            self.btn_apply_ellipsoid["state"] = "normal"
            i = i[0]
            xform = self.fit_objects[i][-2]

            offset_type = self.fit_objects[i][-1]

            pts = self.__get_fit_box_pts(offset_type)

            rel, ste = calimu.pcl_algo.err.get_err(pts, xform)

            set_copyable_text_label(self.lbl_rel_std, f"{rel * 100.0}%")
            set_copyable_text_label(self.lbl_std_err, f"{ste * 100.0}%")

            if rel < 0.005:
                self.lbl_rel_std["bg"] = TKColors.pale_green
            elif rel < 0.05:
                self.lbl_rel_std["bg"] = TKColors.green_yellow
            elif rel < 0.5:
                self.lbl_rel_std["bg"] = TKColors.light_goldenrod
            else:
                self.lbl_rel_std["bg"] = TKColors.pink

            if abs(ste) < 0.01:
                self.lbl_std_err["bg"] = TKColors.pale_green
            elif abs(ste) < 0.1:
                self.lbl_std_err["bg"] = TKColors.green_yellow
            elif abs(ste) < 1:
                self.lbl_std_err["bg"] = TKColors.light_goldenrod
            else:
                self.lbl_std_err["bg"] = TKColors.pink

            set_copyable_text_label(
                self.lbl_ellipsoid_center,
                f"[{xform[0, 3]},\t {xform[1, 3]},\t {xform[2, 3]}]",
            )

            mat_str = f"""
[{xform[0, 0]:.3f},\t {xform[0, 1]:.3f},\t {xform[0, 2]:.3f},\t {xform[0, 3]:.3f}]
[{xform[1, 0]:.3f},\t {xform[1, 1]:.3f},\t {xform[1, 2]:.3f},\t {xform[1, 3]:.3f}]
[{xform[2, 0]:.3f},\t {xform[2, 1]:.3f},\t {xform[2, 2]:.3f},\t {xform[2, 3]:.3f}]
[{xform[3, 0]:.3f},\t {xform[3, 1]:.3f},\t {xform[3, 2]:.3f},\t {xform[3, 3]:.3f}]            
"""
            set_copyable_text_label(self.lbl_ellipsoid_matrix, mat_str)

        else:
            self.btn_apply_ellipsoid["state"] = "disabled"

    def delete_mag_pts_cmd(self):
        (
            locations,
            colors,
            vertices,
        ) = self.display.displayer.callback_instance.get_points()
        indices = []
        # you could search by location instead of color, or keep index info with the points
        # (like get something back from vtk that tells the indices and store that),
        #  but I'm fine with keeping the points one color for now
        for c in range(colors.shape[0]):
            # noinspection PyTypeChecker
            if all(
                    colors[c] == self.store.colors["m"]
            ):  # this isn't a bool pycharm... it's an array of them.
                indices.append(c)
        self.display.displayer.callback_instance.del_points(indices)
        del self.store.mag_points
        self.store.mag_points = []

    def delete_acc_pts_cmd(self):
        (
            locations,
            colors,
            vertices,
        ) = self.display.displayer.callback_instance.get_points()
        indices = []
        # you could search by location instead of color, or keep index info with the points
        # (like get something back from vtk that tells the indices and store that),
        #  but I'm fine with keeping the points one color for now
        for c in range(colors.shape[0]):
            # noinspection PyTypeChecker
            if all(colors[c] == self.store.colors["a"]):
                indices.append(c)
        self.display.displayer.callback_instance.del_points(indices)
        del self.store.acc_points
        self.store.acc_points = []

    def mainloop(self, n: int = 0) -> None:
        t0 = time.time()
        while True:
            try:
                self.update_idletasks()
                self.update()
                self.display.run_once()
                if time.time() - t0 > 1.0 / self.update_fps:
                    vtk_tk_match_height(self.display.displayer.render_window, self)
                    vtk_tk_anchor_left(self.display.displayer.render_window, self)
                    t0 = time.time()

            except KeyboardInterrupt:
                break  # program exited from commandline with ctrl-c
            except tk.TclError:
                break  # pressed x button on tkinter

        self.display.displayer.render_window.Finalize()  # equivalent: renWin.Finalize()
        self.display.displayer.render_window_interactor.TerminateApp()
        self.store.stop()
        self.store.join()

    def __init__(self):
        super().__init__()

        self.fit_objects = []
        self.fit_objects_color_index = 0
        self.ellipsoid_poly = None

        self.title("IMU Setup")

        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.iconbitmap(dir_path + os.sep + "imucal.ico")

        self.update_fps = 60

        self.container_com = None
        self.container_connect = None
        self.container_imu_options = None
        self.container_point_options = None
        self.container_ellipsoid_fit = None
        self.container_ellipsoid_opts = None
        self.container_proj_rtree = None
        self.container_view_opts = None
        self.setup_container_layout()

        bottom = tk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        # endregion
        # region COM CONTAINER
        self.com_var = None
        self.com_menu = None
        self.refresh_button = None
        self.refresh_com()
        # endregion
        # region CONNECT CONTAINER
        self.setup_connect_container()
        # endregion
        # region POINT OPTIONS CONTAINER
        self.setup_point_options()
        # endregion
        # region IMU OPTIONS CONTAINER
        self.setup_imu_options()
        # endregion
        # region ELLIPSOID CONTAINER
        self.fit_list_box = None
        self.setup_ellipsoid_fit()

        self.radio_option_center = None
        self.radio_option_fit = None
        self.radio_option_data = None
        self.setup_ellipsoid_options()

        self.lbl_rel_std = None
        self.lbl_std_err = None
        self.lbl_ellipsoid_center = None
        self.lbl_ellipsoid_matrix = None
        self.btn_apply_ellipsoid = None
        self.no_data = "(No Data)"
        self.setup_ellipsoid_info()
        # endregion
        self.imu = MC6470IMU()
        self.store = IMUPointStore(self.imu)
        self.store.start()
        self.display = IMUPointDisplayer(self.store)
        self.display.displayer.tk_visualize()

    def setup_container_layout(self):
        self.container_com = tk.Frame(self)
        self.container_com.pack(side="top", fill="both")

        self.container_connect = center_packed_frame(self)

        self.container_imu_options = tk.Frame(self)
        self.container_imu_options.pack(side="top", fill="both")

        self.container_point_options = tk.Frame(self)
        self.container_point_options.pack(side="top", fill="both")

        self.container_ellipsoid_fit = tk.Frame(self)
        self.container_ellipsoid_fit.pack(side="top", fill="both")

        self.container_ellipsoid_opts = tk.Frame(self)
        self.container_ellipsoid_opts.pack(side="top", fill="both")

        proj_rtree_container = tk.Frame(self)
        proj_rtree_container.pack(side="top", fill="both")

        self.container_view_opts = tk.Frame(self)
        self.container_view_opts.pack(side="top", fill="both")

        bottom = tk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def setup_connect_container(self):
        c_imu = tk.Button(
            self.container_connect,
            text="Connect IMU",
            height=1,
            command=self.connect_command,
        )
        c_imu.pack(side=tk.LEFT, pady=5)
        d_imu = tk.Button(
            self.container_connect,
            text="Disconnect IMU",
            height=1,
            command=self.disconnect_command,
        )
        d_imu.pack(side=tk.LEFT)

        d_imu["bg"] = TKColors.pink
        c_imu["bg"] = TKColors.green_yellow

    def setup_point_options(self):
        begin_region_with_sep_and_label(self.container_point_options, "Point Options:")

        base_point_opts_container = center_packed_frame(self.container_point_options)

        del_mag = tk.Button(
            base_point_opts_container,
            text="Delete Mag Pts",
            height=1,
            command=self.delete_mag_pts_cmd,
        )
        del_mag.pack(side=tk.LEFT, pady=5)
        del_acc = tk.Button(
            base_point_opts_container,
            text="Delete Accel Pts",
            height=1,
            command=self.delete_acc_pts_cmd,
        )
        del_acc.pack(side=tk.LEFT)

    def setup_imu_options(self):
        begin_region_with_sep_and_label(self.container_imu_options, "IMU Options:")

        imu_ops_container_2 = center_packed_frame(self.container_imu_options)

        d_imu = tk.Button(
            imu_ops_container_2,
            text="Update",
            height=1,
            command=self.display_command,
        )
        d_imu.pack(side=tk.LEFT, pady=5)
        s_imu = tk.Button(
            imu_ops_container_2,
            text="Stop",
            height=1,
            command=self.stop_display_command,
        )
        s_imu.pack(side=tk.LEFT)

        base_imu_opts_container = center_packed_frame(self.container_imu_options)

        g_imu = tk.Button(
            base_imu_opts_container,
            text="Gather",
            height=1,
            command=self.gather_command,
        )
        g_imu.pack(side=tk.LEFT, pady=5)
        s_imu = tk.Button(
            base_imu_opts_container,
            text="Stop",
            height=1,
            command=self.stop_gather_command,
        )
        s_imu.pack(side=tk.LEFT)

    def setup_ellipsoid_fit(self):
        underlined_label(self.container_ellipsoid_fit, "Ellipsoid Fit:")

        self.fit_list_box = tk.Listbox(
            self.container_ellipsoid_fit,
            width=30,
        )
        self.fit_list_box.pack(side=tk.LEFT, expand=True)
        self.fit_list_box.bind("<<ListboxSelect>>", self.fit_list_box_selected)

        fit_option_container = tk.Frame(self.container_ellipsoid_fit)
        fit_option_container.pack(side=tk.LEFT)

        ellipsoid_fit_add_button = tk.Button(
            fit_option_container, text="Add", command=self.add_fit_command
        )
        ellipsoid_fit_add_button.pack(
            side=tk.TOP, anchor=tk.NW, padx=5, expand=True, fill="both"
        )

        ellipsoid_fit_remove_button = tk.Button(
            fit_option_container, text="Remove", command=self.rem_fit_command
        )
        ellipsoid_fit_remove_button.pack(
            side=tk.TOP, anchor=tk.NW, padx=5, expand=True, fill="both"
        )

        ellipsoid_fit_show_button = tk.Button(
            fit_option_container, text="Show", command=self.show_fit_command
        )
        ellipsoid_fit_show_button.pack(
            side=tk.TOP, anchor=tk.NW, padx=5, expand=True, fill="both"
        )

        ellipsoid_fit_hide_button = tk.Button(
            fit_option_container, text="Hide", command=self.hide_fit_command
        )
        ellipsoid_fit_hide_button.pack(
            side=tk.TOP, anchor=tk.NW, padx=5, expand=True, fill="both"
        )

    def setup_ellipsoid_options(self):
        underlined_label(self.container_ellipsoid_opts, "Ellipsoid Options:")

        ellipsoid_opts_subgrid = tk.Frame(self.container_ellipsoid_opts)
        ellipsoid_opts_subgrid.pack(side=tk.TOP)

        self.radio_option_center = tk.IntVar()
        self.radio_option_center.set(4)

        chk8 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Center By Ellipsoid Fit",
            variable=self.radio_option_center,
            value=4,
        )
        chk8.grid(row=0, column=0, sticky=tk.NW)

        chk8 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Center By Sphere Fit",
            variable=self.radio_option_center,
            value=3,
        )
        chk8.grid(row=1, column=0, sticky=tk.NW)
        chk1 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Center By XYZ Bounds",
            variable=self.radio_option_center,
            value=1,
        )
        chk1.grid(row=2, column=0, sticky=tk.NW)

        chk2 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Center By Average",
            variable=self.radio_option_center,
            value=2,
        )
        chk2.grid(row=3, column=0, sticky=tk.NW)

        chk5 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Center At Zero",
            variable=self.radio_option_center,
            value=0,
        )
        chk5.grid(row=4, column=0, sticky=tk.NW)

        self.radio_option_fit = tk.IntVar()
        self.radio_option_fit.set(3)
        chk4 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Fit By Ellipsoid",
            variable=self.radio_option_fit,
            value=3,
        )
        chk4.grid(row=0, column=1, sticky=tk.NW)
        chk4 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Fit By Sphere",
            variable=self.radio_option_fit,
            value=2,
        )
        chk4.grid(row=1, column=1, sticky=tk.NW)
        chk4 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Fit By PCA",
            variable=self.radio_option_fit,
            value=1,
        )
        chk4.grid(row=3, column=1, sticky=tk.NW)

        chk3 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Fit By AABB",
            variable=self.radio_option_fit,
            value=0,
        )
        chk3.grid(row=2, column=1, sticky=tk.NW)

        self.radio_option_data = tk.IntVar()
        chk6 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Fit Magnetometer",
            variable=self.radio_option_data,
            value=0,
        )
        chk6.grid(row=5, column=0, sticky=tk.NW)

        chk7 = tk.Radiobutton(
            ellipsoid_opts_subgrid,
            text="Fit Accelerometer",
            variable=self.radio_option_data,
            value=1,
        )
        chk7.grid(row=6, column=0, sticky=tk.NW)

    def setup_ellipsoid_info(self):
        underlined_label(self.container_ellipsoid_opts, text="Selected Ellipsoid Info:")

        ellipsoid_rel_std_container = center_packed_frame(self.container_ellipsoid_opts)

        lbl_err = tk.Label(ellipsoid_rel_std_container, text="RSD For Ellipsoid:")
        lbl_err.pack(side=tk.LEFT)

        self.lbl_rel_std = selectable_label(
            ellipsoid_rel_std_container, self.no_data, width=20
        )
        self.lbl_rel_std["bg"] = TKColors.pink
        self.lbl_rel_std.pack(side=tk.LEFT, pady=5)

        ellipsoid_std_err_container = center_packed_frame(self.container_ellipsoid_opts)

        lbl_err = tk.Label(ellipsoid_std_err_container, text="MAE For Ellipsoid:")
        lbl_err.pack(side=tk.LEFT)

        self.lbl_std_err = selectable_label(
            ellipsoid_std_err_container, self.no_data, width=20
        )
        self.lbl_std_err["bg"] = TKColors.pink
        self.lbl_std_err.pack(side=tk.LEFT, pady=5)

        ellipsoid_center_container = center_packed_frame(self.container_ellipsoid_opts)

        lbl_center = tk.Label(ellipsoid_center_container, text="Ellipsoid Center:")
        lbl_center.pack(side=tk.LEFT)

        self.lbl_ellipsoid_center = selectable_label(
            ellipsoid_center_container, self.no_data, width=20, height=4
        )
        self.lbl_ellipsoid_center.pack(side=tk.LEFT, pady=5)

        ellipsoid_mat_container = tk.Frame(self.container_ellipsoid_opts)
        ellipsoid_mat_container.pack(side="top", fill="both")

        lbl_center = tk.Label(ellipsoid_mat_container, text="Ellipsoid Matrix:")
        lbl_center.pack(side=tk.TOP)

        self.lbl_ellipsoid_matrix = selectable_label(
            ellipsoid_mat_container, self.no_data, height=6, width=40
        )
        tmp_fnt = self.lbl_ellipsoid_matrix["font"]
        if isinstance(tmp_fnt, str):
            tmp_fnt = (tmp_fnt, 8)
        else:
            raise TypeError("Unexpected type returned from tkinter font")
        self.lbl_ellipsoid_matrix["font"] = tmp_fnt
        self.lbl_ellipsoid_matrix.pack(side=tk.TOP, pady=5)

        self.btn_apply_ellipsoid = tk.Button(
            self.container_ellipsoid_opts,
            text="APPLY SELECTED ELLIPSOID OFFSETS",
            command=self.set_offsets,
        )
        self.btn_apply_ellipsoid.pack(side=tk.TOP, pady=5)
        self.btn_apply_ellipsoid["state"] = "disabled"
        self.btn_apply_ellipsoid["bg"] = "light blue"


def main():
    testObj = IMUApp()
    testObj.mainloop()


if __name__ == "__main__":
    main()
