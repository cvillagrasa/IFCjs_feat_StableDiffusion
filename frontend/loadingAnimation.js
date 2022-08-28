document.write(`
    <div id="loading-animation">
        <svg class="gegga">
            <defs>
                <filter id="gegga">
                    <feGaussianBlur in="SourceGraphic" stdDeviation="7" result="blur"></feGaussianBlur>
                    <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 20 -10" result="inreGegga"></feColorMatrix>
                    <feComposite in="SourceGraphic" in2="inreGegga" operator="atop"></feComposite>
                </filter>
            </defs>
        </svg>
        <svg class="snurra">
            <defs>
                <linearGradient id="linjärGradient">
                    <stop class="stopp1" offset="0"></stop>
                    <stop class="stopp2" offset="1"></stop>
                </linearGradient>
                <linearGradient y2="160" x2="160" y1="40" x1="40" gradientUnits="userSpaceOnUse" id="gradient" xlink:href="#linjärGradient"></linearGradient>
            </defs>
            <circle class="strecken" cx="8vh" cy="8vh" r="64"></circle>
        </svg>
    </div>
`);