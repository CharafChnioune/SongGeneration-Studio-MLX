module.exports = {
  version: "4.0",
  run: [
    {
      method: "fs.rm",
      params: {
        path: "env"
      }
    }
  ]
}
