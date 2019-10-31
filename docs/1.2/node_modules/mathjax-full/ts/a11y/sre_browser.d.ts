declare namespace sre {

  export type colorType = {color: string, alpha: number};
  export type colorString = {foreground: string, background: string};

  interface SpeechGenerator {
    generateSpeech(node: HTMLElement, xml: HTMLElement): void;
    setOptions(options: Object): void;
    getOptions(): {[key: string]: string};
  }

  interface Highlighter {
    highlight(nodes: Node[]): void;
    unhighlight(): void;
    highlightAll(node: Node): void;
    unhighlightAll(node: Node): void;
    colorString(): colorString;
    isMactionNode(node: Node): boolean;
    colorizeAll(node: Node): void;
    uncolorizeAll(node: Node): void;
  }

  interface Focus {
    getNodes(): Node[];
  }

  interface Walker {
    activate(): void;
    deactivate(): void;
    speech(): string;
    move(key: number): boolean;
    getFocus(update?: boolean): Focus;
    update(options: {[key: string]: string}): void;
  }

}

declare namespace sre.WalkerFactory {
  export function walker(kind: string,
                         node: Node,
                         generator: SpeechGenerator,
                         highlighter: Highlighter,
                         mml: Node): Walker;
}

declare namespace sre.SpeechGeneratorFactory {
  export function generator(kind: string): sre.SpeechGenerator;
}

declare namespace sre.Engine {
  export function isReady(): boolean;
}

declare namespace sre.HighlighterFactory {

  export function highlighter(fore: colorType,
                              back: colorType,
                              info: {renderer: string, browser?: string}
                             ): Highlighter;

}

declare namespace sre.ClearspeakPreferences {

  export function smartPreferences(item: Object, locale: string): string[];

}
