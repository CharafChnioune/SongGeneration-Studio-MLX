module.exports = {
  version: "4.0",
  daemon: true,
  run: [
    {
      method: "local.set",
      params: {
        port: "{{port}}",
        url: "{{'http://127.0.0.1:' + port}}"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "python main.py --host 127.0.0.1 --port {{local.port}}",
        on: [
          {
            event: "/http:\\/\\/127\\.0\\.0\\.1:\\d+/",
            done: true
          }
        ]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event ? input.event[0] : local.url}}"
      }
    },
    {
      method: "log",
      params: {
        raw: "SongGeneration Studio running at {{local.url}}"
      }
    }
  ]
}
