var state = {
    connected: false,
    page: "options",
    passes: [],
    documents: [],
    clipscrolltree: [],
}

class Connection {
    constructor() {
        this.ws = null;
    }

    connect() {
        var ws = new WebSocket("ws://127.0.0.1:3583");

        ws.onopen = function() {
            state.connected = true;
            state.page = "options";
        }

        ws.onmessage = function(evt) {
            var json = JSON.parse(evt.data);
            if (json['kind'] == "passes") {
                state.passes = json['passes'];
            } else if (json['kind'] == "documents") {
                state.documents = json['root'];
            } else if (json['kind'] == "clipscrolltree") {
                state.clipscrolltree = json['root'];
            } else {
                console.warn("unknown message kind: " + json['kind']);
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
            <div v-if="state.connected" class="section">
                <div class="container">
                    <div class="columns">
                        <div class="column is-3">
                            <mainmenu :page=state.page></mainmenu>
                        </div>
                        <div class="column">
                            <options v-if="state.page == 'options'"></options>
                            <passview v-if="state.page == 'passes'" :passes=state.passes></passview>
                            <documentview v-if="state.page == 'documents'" :documents=state.documents></documentview>
                            <clipscrolltreeview v-if="state.page == 'clipscrolltree'" :clipscrolltree=state.clipscrolltree></documentview>
                        </div>
                    </div>
                </div>
            </div>
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
        },
        setAlphaRectsDebugger(enabled) {
            if (enabled) {
                connection.send("enable_alpha_rects_debug");
            } else {
                connection.send("disable_alpha_rects_debug");
            }
        }
    },
    template: `
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
            <div class="field">
                <label class="checkbox">
                    <input type="checkbox" v-on:click="setAlphaRectsDebugger($event.target.checked)">
                    Alpha primitive rects debugger
                </label>
            </div>
        </div>
    `
})

Vue.component('passview', {
    props: [
        'passes'
    ],
    methods: {
        fetch: function() {
            connection.send("fetch_passes");
        }
    },
    template: `
        <div class="box">
            <h1 class="title">Passes <a v-on:click="fetch" class="button is-info">Refresh</a></h1>
            <hr/>
            <div v-for="(pass, pass_index) in passes">
                <p class="has-text-black-bis">Pass {{pass_index}}</p>
                <div v-for="(target, target_index) in pass.targets">
                    <p style="text-indent: 2em;" class="has-text-grey-dark">Target {{target_index}} ({{target.kind}})</p>
                    <div v-for="(batch, batch_index) in target.batches">
                        <p style="text-indent: 4em;" class="has-text-grey">Batch {{batch_index}} ({{batch.description}}, {{batch.kind}}, {{batch.count}} instances)</p>
                    </div>
                </div>
                <hr/>
            </div>
        </div>
    `
})

Vue.component('treeview', {
    props: {
        model: Object
    },
    data: function () {
        return {
            open: false
        }
    },
    computed: {
        isFolder: function () {
            return this.model.children && this.model.children.length
        }
    },
    methods: {
        toggle: function () {
            if (this.isFolder) {
                this.open = !this.open
            }
        },
    },
    template: `
        <li>
            <div v-on:click="toggle">
                <span v-if="isFolder">[{{open ? '-' : '+'}}]</span>
                {{model.description}}
            </div>
            <ul style="padding-left: 1em; line-height: 1.5em;" v-show="open" v-if="isFolder">
                <treeview v-for="model in model.children" :model="model"></treeview>
            </ul>
        </li>
    `
})

Vue.component('documentview', {
    props: [
        'documents'
    ],
    methods: {
        fetch: function() {
            connection.send("fetch_documents");
        }
    },
    template: `
        <div class="box">
            <h1 class="title">Documents <a v-on:click="fetch" class="button is-info">Refresh</a></h1>
            <hr/>
            <div>
                <ul>
                    <treeview :model=documents></treeview>
                </ul>
            </div>
        </div>
    `
})

Vue.component('clipscrolltreeview', {
    props: [
        'clipscrolltree'
    ],
    methods: {
        fetch: function() {
            connection.send("fetch_clipscrolltree");
        }
    },
    template: `
        <div class="box">
            <h1 class="title">Clip-scroll Tree <a v-on:click="fetch" class="button is-info">Refresh</a></h1>
            <hr/>
            <div>
                <ul>
                    <treeview :model=clipscrolltree></treeview>
                </ul>
            </div>
        </div>
    `
})

Vue.component('mainmenu', {
    props: [
        'page',
    ],
    methods: {
        setPage: function(id) {
            state.page = id;
        }
    },
    template: `
        <aside class="menu">
            <p class="menu-label">
                Pages
            </p>
            <ul class="menu-list">
                <li><a v-on:click="setPage('options')" v-bind:class="{ 'is-active': page == 'options' }">Debug Options</a></li>
                <li><a v-on:click="setPage('passes')" v-bind:class="{ 'is-active': page == 'passes' }">Passes</a></li>
                <li><a v-on:click="setPage('documents')" v-bind:class="{ 'is-active': page == 'documents' }">Documents</a></li>
                <li><a v-on:click="setPage('clipscrolltree')" v-bind:class="{ 'is-active': page == 'clipscrolltree' }">Clip-scroll Tree</a></li>
            </ul>
        </aside>
    `
})

new Vue({
    el: '#app',
    data: {
        state: state,
    },
    template: "<app :state=state></app>",
})
