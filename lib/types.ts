export type TokenThought = {
  token: string;
  thoughts: string[];
};

export type ThoughtResult = {
  answer: string;
  synthesis: string;
  tokens: TokenThought[];
};
