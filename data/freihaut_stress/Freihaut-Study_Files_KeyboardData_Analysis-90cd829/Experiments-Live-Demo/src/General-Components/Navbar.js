// React Imports
import React from 'react';

export default function Navbar(props) {
        return(
            <nav className="navbar is-info" style={{width: "100%"}}>
                    <div className="navbar-start">
                        <div className="navbar-item">
                            <div className="buttons">
                                <button className="button is-light" onClick={() => props.cancelExp()}>
                                    <strong>Exit Experiment</strong>
                                </button>
                            </div>
                        </div>

                    </div>
                <div className="navbar-menu">
                    <div className="navbar-end">
                        <div className="navbar-item">
                            <div className="buttons">
                                <button className="button is-light" onClick={() => props.goBackPhase()}
                                        disabled={props.skipButtonDisabled}>
                                    <strong>Previous Page</strong>
                                </button>
                                <button className="button is-light" onClick={() => props.proceedPhase()}
                                        disabled={props.skipButtonDisabled}>
                                    <strong>Next Page</strong>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </nav>
        )
}