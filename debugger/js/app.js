var state = {
    connected: false,
    batches: [],
}

class Connection {
    constructor() {
        this.ws = null;
    }

    connect() {
        var ws = new WebSocket("ws://127.0.0.1:3583");

        ws.onopen = function() {
            state.connected = true;
        }

        ws.onmessage = function(evt) {
            var json = JSON.parse(evt.data);
            if (json['kind'] == "batches") {
                state.batches = json['batches'];
            }
        }

        ws.onclose = function() {
            state.connected = false;
        }

        this.ws = ws;
    }

    send(msg) {
        if (this.ws !== null) {
            this.ws.send(msg);
        }
    }

    disconnect() {
        if (this.ws !== null) {
            this.ws.close();
            this.ws = null;
        }
    }
}

var connection = new Connection();

Vue.component('app', {
    props: [
        'state'
    ],
    template: `
        <div>
            <navbar :connected=state.connected></navbar>
            <options v-if="state.connected"></options>
            <batchview v-if="state.connected" :batches=state.batches></batchview>
        </div>
    `
})

Vue.component('navbar', {
    props: [
        'connected'
    ],
    methods: {
        connect() {
            connection.connect();
        },
        disconnect() {
            connection.disconnect();
        },
    },
    template: `
      <nav class="navbar has-shadow">
      <div class="navbar-brand">
        <a class="navbar-item" href="#">WebRender Debugger</a>
      </div>

      <div class="navbar-menu">
        <div class="navbar-start">
        </div>

        <div class="navbar-end">
          <div class="navbar-item">
              <p class="control">
                <button v-if="connected" v-on:click="disconnect" class="button is-danger">Disconnect</button>
                <button v-else v-on:click="connect" class="button is-success">Connect</button>
              </p>
            </div>
          </div>
        </div>
      </div>
    </nav>
    `
})

Vue.component('options', {
    methods: {
        setProfiler(enabled) {
            if (enabled) {
                connection.send("enable_profiler");
            } else {
                connection.send("disable_profiler");
            }
        },
        setTextureCacheDebugger(enabled) {
            if (enabled) {
                connection.send("enable_texture_cache_debug");
            } else {
                connection.send("disable_texture_cache_debug");
            }
        },
        setRenderTargetDebugger(enabled) {
            if (enabled) {
                connection.send("enable_render_target_debug");
            } else {
                connection.send("disable_render_target_debug");
            }
        }
    },
    template: `
        <div class="section">
            <div class="container">
                <div class="box">
                    <div class="field">
                        <label class="checkbox">
                            <input type="checkbox" v-on:click="setProfiler($event.target.checked)">
                            Profiler
                        </label>
                    </div>
                    <div class="field">
                        <label class="checkbox">
                            <input type="checkbox" v-on:click="setTextureCacheDebugger($event.target.checked)">
                            Texture cache debugger
                        </label>
                    </div>
                    <div class="field">
                        <label class="checkbox">
                            <input type="checkbox" v-on:click="setRenderTargetDebugger($event.target.checked)">
                            Render target debugger
                        </label>
                    </div>
                </div>
            </div>
        </div>
    `
})

Vue.component('batchview', {
    props: [
        'batches'
    ],
    methods: {
        fetch: function() {
            connection.send("fetch_batches");
        }
    },
    template: `
        <div class="container">
            <div class="box">
                <h1 class="title">Batches <a v-on:click="fetch" class="button is-info">Refresh</a></h1>
                <hr/>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Batch Kind</th>
                            <th>Instances</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="batch in batches">
                            <td>{{ batch.kind }}</td>
                            <td>{{ batch.count }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `
})

new Vue({
    el: '#app',
    data: {
        state: state,
    },
    template: "<app :state=state></app>",
})
