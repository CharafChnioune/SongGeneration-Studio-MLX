module.exports = {
  version: "2.0",
  title: "SongGeneration Studio MLX",
  description: "MLX-only song generation studio with lyrics+style AI and auto model downloads.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installed = info.exists("env");
    const running = info.running("start.js");
    const local = info.local("start.js") || {};
    const url = local.url;

    if (!installed) {
      return [
        { icon: "fa-solid fa-plug", text: "Install", href: "install.js", default: true },
        { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" }
      ];
    }

    if (running) {
      return [
        { icon: "fa-solid fa-rocket", text: "Web UI", href: url || "start.js" },
        { icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" },
        { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
        { icon: "fa-solid fa-broom", text: "Factory Reset", href: "reset.js" }
      ];
    }

    return [
      { icon: "fa-solid fa-power-off", text: "Start", href: "start.js", default: true },
      { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
      { icon: "fa-solid fa-broom", text: "Factory Reset", href: "reset.js" }
    ];
  }
}
